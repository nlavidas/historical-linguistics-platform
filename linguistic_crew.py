
from crewai import Agent, Task, Crew, Process
from langchain_community.llms import Ollama

# Initialize Ollama LLM
llm = Ollama(model="qwen2.5-coder:7b", base_url="http://localhost:11434")

# Define Agents
archivist = Agent(
    role="Data Archivist",
    goal="Collect and organize historical texts from various sources",
    backstory="Expert in digital archives and text curation",
    llm=llm,
    verbose=True
)

philologist = Agent(
    role="Linguistic Analyst",
    goal="Analyze, lemmatize, and parse historical texts",
    backstory="Classical philologist with expertise in Ancient Greek and Latin",
    llm=llm,
    verbose=True
)

curator = Agent(
    role="Output Curator",
    goal="Format and export analyzed texts in standard formats",
    backstory="Digital humanities specialist focused on TEI-XML and standards",
    llm=llm,
    verbose=True
)

# Define Tasks
collection_task = Task(
    description="Scan repositories and collect new historical texts",
    agent=archivist,
    expected_output="List of collected texts with metadata"
)

analysis_task = Task(
    description="Lemmatize and parse collected texts using Stanza and CLTK",
    agent=philologist,
    expected_output="Annotated texts with morphological and syntactic analysis"
)

export_task = Task(
    description="Export analyzed texts in TEI-XML and CoNLL-U formats",
    agent=curator,
    expected_output="Formatted output files ready for research"
)

# Create Crew
linguistic_crew = Crew(
    agents=[archivist, philologist, curator],
    tasks=[collection_task, analysis_task, export_task],
    process=Process.sequential,
    verbose=True
)

if __name__ == "__main__":
    result = linguistic_crew.kickoff()
    print(result)
