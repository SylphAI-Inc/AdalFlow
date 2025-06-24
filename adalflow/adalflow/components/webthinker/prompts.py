"""WebThinker trainable prompt templates."""

from adalflow.optim.parameter import Parameter
from adalflow.core.component import Component
from typing import Dict, Any

class WebThinkerPrompts(Component):
    """Trainable prompt templates for WebThinker components."""
    
    def __init__(self):
        super().__init__()
        
        # System prompt for WebThinker
        self.system_prompt = Parameter(
            data="""You are a reasoning assistant with the ability to perform web searches to help you answer the user's question accurately. You have special tools:

- To perform a search: write <|begin_search_query|>your query here<|end_search_query|>.
Then, the system will search and analyze relevant web pages, then provide you with helpful information in the format <|begin_search_result|> ...search results... <|end_search_result|>.

You can repeat the search process multiple times if necessary. Once you have all the information you need, continue your reasoning.

Remember:
- Use <|begin_search_query|> to request a web search and end with <|end_search_query|>.
- When done searching, continue your reasoning.
- Think step by step to solve the problem.""",
            requires_opt=True,
            role_desc="System prompt for WebThinker agent"
        )
        
        # Search query generation prompt
        self.search_query_prompt = Parameter(
            data="""Based on the current reasoning context and the original question, generate a specific search query that would help gather the missing information needed to answer the question.

Current reasoning context:
{reasoning_context}

Original question:
{question}

Generate a focused search query that addresses the key information gap:""",
            requires_opt=True,
            role_desc="Prompt for generating search queries"
        )
        
        # Search intent analysis prompt
        self.intent_analysis_prompt = Parameter(
            data="""Based on the previous reasoning steps and the search query, provide a detailed analysis of what specific information is being sought.

Previous reasoning:
{reasoning_context}

Search query:
{search_query}

Provide the detailed search intent - what exactly are we looking for?""",
            requires_opt=True,
            role_desc="Prompt for analyzing search intent"
        )
        
        # Content analysis prompt
        self.content_analysis_prompt = Parameter(
            data="""You are analyzing web content to extract information relevant to a specific search query and intent.

Search query: {search_query}
Search intent: {search_intent}

Web content:
{content}

Extract all relevant information that addresses the search query and intent. Focus on factual, accurate information. Format the output clearly and concisely:""",
            requires_opt=True,
            role_desc="Prompt for analyzing web content"
        )
        
        # Information integration prompt
        self.integration_prompt = Parameter(
            data="""Integrate the new information from web search into the ongoing reasoning process.

Previous reasoning:
{previous_reasoning}

New information from search:
{new_information}

Original question:
{question}

Continue your reasoning by incorporating this new information. Build upon what you already know:""",
            requires_opt=True,
            role_desc="Prompt for integrating new information"
        )
        
        # Final answer synthesis prompt
        self.synthesis_prompt = Parameter(
            data="""Based on all the reasoning and information gathered, provide a comprehensive answer to the question.

Complete reasoning chain:
{reasoning_chain}

Original question:
{question}

Provide your final answer, ensuring it directly addresses the question and is supported by the evidence gathered:""",
            requires_opt=True,
            role_desc="Prompt for synthesizing final answer"
        )
    
    def get_search_query_prompt(self, reasoning_context: str, question: str) -> str:
        """Get formatted search query generation prompt."""
        return self.search_query_prompt.data.format(
            reasoning_context=reasoning_context,
            question=question
        )
    
    def get_intent_analysis_prompt(self, reasoning_context: str, search_query: str) -> str:
        """Get formatted intent analysis prompt."""
        return self.intent_analysis_prompt.data.format(
            reasoning_context=reasoning_context,
            search_query=search_query
        )
    
    def get_content_analysis_prompt(self, search_query: str, search_intent: str, content: str) -> str:
        """Get formatted content analysis prompt."""
        return self.content_analysis_prompt.data.format(
            search_query=search_query,
            search_intent=search_intent,
            content=content
        )
    
    def get_integration_prompt(self, previous_reasoning: str, new_information: str, question: str) -> str:
        """Get formatted integration prompt."""
        return self.integration_prompt.data.format(
            previous_reasoning=previous_reasoning,
            new_information=new_information,
            question=question
        )
    
    def get_synthesis_prompt(self, reasoning_chain: str, question: str) -> str:
        """Get formatted synthesis prompt."""
        return self.synthesis_prompt.data.format(
            reasoning_chain=reasoning_chain,
            question=question
        )
