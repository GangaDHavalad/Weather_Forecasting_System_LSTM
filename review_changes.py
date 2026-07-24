import streamlit as st
import subprocess
import shlex


def run(cmd):
    try:
        result = subprocess.run(shlex.split(cmd), capture_output=True, text=True, check=False)
        return result.returncode, result.stdout + result.stderr
    except Exception as e:
        return 1, str(e)


st.set_page_config(page_title="Git Review UI", layout="wide")
st.title("Git Review — Allow or Skip Changes")

st.markdown(
    "Use this page to review local git changes and either 'Allow' (stage → commit → push) or 'Skip' (restore)."
)

code, status = run("git rev-parse --abbrev-ref HEAD")
branch = status.strip() if code == 0 else "(unknown)"
st.write(f"**Current branch:** {branch}")

st.header("Git Status")
code, status = run("git status --porcelain")
if status.strip() == "":
    st.success("Working directory clean — no changes detected.")
else:
    st.code(status)

st.header("Diff (modified files)")
code, diff = run("git --no-pager diff")
if diff.strip() == "":
    st.info("No diffs for tracked files. Untracked files may still exist.")
else:
    st.code(diff)

col1, col2 = st.columns(2)

with col1:
    st.markdown("<div style='background:#e6ffed;padding:10px;border-radius:6px'>", unsafe_allow_html=True)
    st.write("### ✅ Allow changes")
    commit_msg = st.text_input("Commit message", value="Make model loading lazy; update requirements")
    if st.button("Allow (stage → commit → push)"):
        if commit_msg.strip() == "":
            st.error("Commit message cannot be empty.")
        else:
            st.info("Staging all changes...")
            c, out = run("git add -A")
            st.code(out)
            st.info("Committing...")
            c, out = run(f"git commit -m \"{commit_msg}\"")
            st.code(out)
            st.info("Pushing to origin...")
            c, out = run(f"git push -u origin {branch}")
            if c == 0:
                st.success("Pushed successfully.")
                st.code(out)
            else:
                st.error("Push failed — check output below.")
                st.code(out)
    st.markdown("</div>", unsafe_allow_html=True)

with col2:
    st.markdown("<div style='background:#ffecec;padding:10px;border-radius:6px'>", unsafe_allow_html=True)
    st.write("### ❌ Skip changes")
    st.write("This will discard local modifications and remove untracked files — use with care.")
    if st.button("Skip (restore working tree)"):
        confirm = st.checkbox("I understand this will discard local changes and untracked files", key="confirm_skip")
        if confirm:
            st.info("Restoring tracked files...")
            c, out = run("git restore .")
            st.code(out)
            st.info("Removing untracked files and directories...")
            c, out = run("git clean -fd")
            st.code(out)
            st.success("Working tree restored.")
        else:
            st.warning("Please confirm the checkbox to proceed.")
    st.markdown("</div>", unsafe_allow_html=True)

st.markdown("---")
if st.button("Refresh status"):
    st.experimental_rerun()
