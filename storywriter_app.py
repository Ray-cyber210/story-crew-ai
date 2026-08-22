import os
import streamlit as st
from crewai import Agent, Task, Crew, Process, LLM

st.set_page_config(page_title="AI Story Crew", page_icon="📖", layout="wide")

# ---- the key comes from Streamlit secrets, never from the code ----
try:
    os.environ["GROQ_API_KEY"] = st.secrets["GROQ_API_KEY"]
except (KeyError, FileNotFoundError):
    st.error("No API key found. Add GROQ_API_KEY in Settings -> Secrets.")
    st.stop()

llm = LLM(model="groq/llama-3.3-70b-versatile", temperature=1)

MAX_RUNS = 5

GENRES = {
    "Fantasy":       "You love magic systems, quests and mythical creatures.",
    "Mystery":       "You love clues, red herrings and a satisfying reveal.",
    "Sci-Fi":        "You love future tech, space and big what-if questions.",
    "Horror":        "You love slow dread, unease and a genuinely unsettling turn.",
    "Comedy":        "You love banter, timing and absurd situations that escalate.",
}

LENGTHS = {
    "Flash (short)":  "Keep the draft to 3-4 short paragraphs. Every sentence must earn its place.",
    "Standard":       "Write a draft of roughly 6-8 paragraphs with room to breathe.",
    "Extended":       "Write a fuller draft of roughly 10-12 paragraphs with more scene detail.",
}


def build_crew(idea, genre, length):
    plotter = Agent(
        role="Story Plotter",
        goal="Turn a simple story idea into a clear outline with a beginning, middle, and end, plus the key turning points",
        backstory=(
            "You are a meticulous story architect. You think in structure "
            "before you think in sentences - setup, conflict, climax, "
            "resolution. You never write full prose yourself; you hand "
            "other writers a skeleton they can't get lost inside. "
            + GENRES[genre]
        ),
        llm=llm, verbose=False, allow_delegation=False, max_iter=3,
    )

    writer = Agent(
        role="Story Writer",
        goal="Turn a story outline into a vivid, well-paced first draft",
        backstory=(
            "You are an imaginative fiction writer with a strong sense of "
            "voice and pacing. You take someone else's outline and bring "
            "it to life with scenes, dialogue, and description - you "
            "never skip ahead or ignore beats the outline set up. "
            + LENGTHS[length]
        ),
        llm=llm, verbose=False, allow_delegation=False, max_iter=3,
    )

    editor = Agent(
        role="Story Editor",
        goal="Cut the draft down to a tight, punchy final version without losing the core story",
        backstory=(
            "You are a ruthless but fair editor. You cut filler, tighten "
            "sentences, and remove anything that doesn't earn its place - "
            "but you never cut a plot point the story actually needs. "
            "Your motto: if it doesn't move the story forward, it goes."
        ),
        llm=llm, verbose=False, allow_delegation=False, max_iter=3,
    )

    plot_task = Task(
        description=(
            f"Turn this story idea into a clear outline: '{idea}'. "
            f"Genre: {genre}. Give a beginning, middle, and end, plus 2-3 "
            f"key turning points. Keep it as a structured outline, not prose."
        ),
        expected_output="A structured outline with beginning, middle, end, and key turning points.",
        agent=plotter,
    )

    draft_task = Task(
        description=(
            f"Using the outline, write a first draft of the story. "
            f"Genre: {genre}. {LENGTHS[length]}"
        ),
        expected_output="A full first draft of the story in prose.",
        agent=writer,
        context=[plot_task],
    )

    edit_task = Task(
        description=(
            "Using the draft, cut it down into a tight, polished final "
            "version. Remove filler and tighten sentences, but keep every "
            "plot point the story needs."
        ),
        expected_output="A tightened, polished final version of the story.",
        agent=editor,
        context=[draft_task],
    )

    return plotter, writer, editor, plot_task, draft_task, edit_task


def run_crew(agents, tasks):
    return str(Crew(agents=agents, tasks=tasks,
                    process=Process.sequential, verbose=False).kickoff())


for k, v in [("runs", 0), ("story", None)]:
    if k not in st.session_state:
        st.session_state[k] = v

# ---- sidebar ----
with st.sidebar:
    st.markdown("### 👥 Your crew")
    st.markdown("**🧩 Plotter** — builds the outline\n\n"
                "**✍️ Writer** — drafts the story\n\n"
                "**✂️ Editor** — cuts it down")
    st.divider()
    genre = st.selectbox("📚 Genre", list(GENRES))
    st.caption(GENRES[genre])
    length = st.radio("📏 Draft length", list(LENGTHS))
    st.caption(LENGTHS[length])
    st.divider()
    st.markdown(f"**Runs left:** {MAX_RUNS - st.session_state.runs}")
    if st.session_state.story and st.button("🔄 Start over", use_container_width=True):
        st.session_state.story = None
        st.rerun()

# ---- main ----
st.title("📖 AI Story Crew")
st.caption("Three AI agents turn your idea into a plotted, drafted, edited story.")

idea = st.text_input(
    "What's your story idea?",
    placeholder="e.g. A lighthouse keeper who finds a message that hasn't been written yet",
    max_chars=150,
)

if st.button("🚀 Write my story", type="primary", use_container_width=True):
    if st.session_state.runs >= MAX_RUNS:
        st.error("You have used all your runs. Refresh the page to reset.")
    elif len(idea.strip()) < 5:
        st.warning("Type a story idea first.")
    else:
        i = idea.strip()
        try:
            plotter, writer, editor, plot_task, draft_task, edit_task = build_crew(i, genre, length)

            with st.status("🧩 Plotter is building the outline...") as s:
                outline = run_crew([plotter], [plot_task])
                s.update(label="🧩 Outline ready", state="complete")

            with st.status("✍️ Writer is drafting the story...") as s:
                draft = run_crew([writer], [draft_task])
                s.update(label="✍️ Draft ready", state="complete")

            with st.status("✂️ Editor is tightening the final cut...") as s:
                final = run_crew([editor], [edit_task])
                s.update(label="✂️ Final version ready", state="complete")

            st.session_state.runs += 1
            st.session_state.story = {
                "idea": i, "genre": genre, "length": length,
                "outline": outline, "draft": draft, "final": final,
            }
            st.rerun()

        except Exception as e:
            st.error("Something went wrong.")
            st.caption(f"{type(e).__name__}: {e}")

s = st.session_state.story
if s:
    st.divider()
    st.subheader(f"📖 {s['idea']} · {s['genre']} · {s['length']}")

    tab1, tab2, tab3 = st.tabs(["✂️ Final story", "✍️ Full draft", "🧩 Outline"])
    with tab1:
        st.markdown(s["final"])
    with tab2:
        st.markdown(s["draft"])
    with tab3:
        st.markdown(s["outline"])

    st.download_button(
        "⬇️ Download the story",
        data=(f"STORY: {s['idea']} ({s['genre']}, {s['length']})\n\n"
              f"=== FINAL VERSION ===\n{s['final']}\n\n"
              f"=== FULL DRAFT ===\n{s['draft']}\n\n"
              f"=== OUTLINE ===\n{s['outline']}"),
        file_name=f"story_{s['genre'].lower().replace(' ', '_')}.txt",
        mime="text/plain",
    )
