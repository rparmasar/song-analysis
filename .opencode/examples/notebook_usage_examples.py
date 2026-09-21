"""
Example usage of the Jupyter Notebook Reading Skill

This script demonstrates both invocation patterns:
1. Agent mode: task(subagent_type='jupyter-notebook-referencer', ...)
2. Skill mode: skill(name='jupyter-notebook-referencer')
"""

from song_analysis.types import TrackAttributeColumns


def use_agent_mode():
    """
    Agent mode for complex multi-step analysis tasks.
    
    Usage:
        task(
            subagent_type='jupyter-notebook-referencer',
            prompt='Summarize the EDA methodology from main.ipynb'
        )
    """
    print("=== AGENT MODE EXAMPLE ===")
    
    # Example prompts for different use cases:
    prompts = [
        "Summarize the EDA methodology from main.ipynb",
        "What were the key findings from the correlation analysis?",
        "Explain why LightGBM was selected as best model",
        "What are the top 5 features most predictive of popularity?"
    ]
    
    for prompt in prompts:
        print(f"\nPrompt: {prompt}")
        # In production, you would call:
        # task(subagent_type='jupyter-notebook-referencer', prompt=prompt)


def use_skill_mode():
    """
    Skill mode for quick lookups and context injection.
    
    Usage:
        skill(name='jupyter-notebook-referencer')
        # Then read specific sections on demand
    """
    print("\n=== SKILL MODE EXAMPLE ===")
    
    print("""
After activating skill:

skill(name='jupyter-notebook-referencer')

You can then:

1. Read specific sections:
   - "Show me the correlation findings from main.ipynb"
   - "What did model_evaluation.ipynb say about grid search results?"
   
2. Cross-reference notebooks:
   - "Compare the feature lists between main.ipynb and model_evaluation.ipynb"
   
3. Get summaries:
   - "Summarize the top 5 findings from scoring_dancehall_tracks.ipynb"

""")


def read_notebook_sections():
    """
    Read specific sections of notebooks using incremental approach.
    
    This respects the 24K context limit by reading in chunks.
    """
    print("=== INCREMENTAL READING EXAMPLE ===")
    
    sections = {
        "main.ipynb": [
            "response distribution analysis",
            "correlation findings", 
            "feature redundancy notes"
        ],
        "model_evaluation.ipynb": [
            "train/test split details",
            "grid search results",
            "RMSE vs R2 justification"
        ],
        "scoring_dancehall_tracks.ipynb": [
            "top/lower predictions",
            "PDP analysis findings"
        ]
    }
    
    for notebook, sections_list in sections.items():
        print(f"\n{notebook}:")
        for section in sections_list:
            # In production: read_notebook(notebook, section=section)
            print(f"  - {section}")


def cross_reference_notebooks():
    """
    Cross-reference information across multiple notebooks.
    """
    print("\n=== CROSS-REFERENCE EXAMPLE ===")
    
    questions = [
        "What feature engineering was done (from model_evaluation)?",
        "Which features were dropped and why? (main.ipynb + model_evaluation)",
        "How do the training data characteristics relate to scoring results?"
    ]
    
    for question in questions:
        print(f"\nCross-reference query: {question}")


if __name__ == "__main__":
    use_agent_mode()
    use_skill_mode()
    read_notebook_sections()
    cross_reference_notebooks()
    
    print("\n" + "=" * 60)
    print("Skill is ready to use!")
    print("=" * 60)
