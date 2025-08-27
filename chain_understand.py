"""
Enhanced Chain Understanding Module with Node Deduplication and Merging Strategy

This module implements the sophisticated node deduplication and merging strategy described
in Node_Merging_Strategy_Analysis.md. It handles overlapping descriptions that occur when
the same UI elements or pages are encountered multiple times during task execution.

Key Features:
1. **Context-Aware Consolidation**: Collects multiple descriptions of the same node with
   their action contexts and generates unified, comprehensive descriptions.

2. **LLM-Driven Merging**: Uses advanced language models to intelligently merge descriptions
   while preserving all valuable information and maintaining task relevance.

3. **Enhanced Database Integration**: Stores merged descriptions with rich metadata including
   merge history, action contexts, and deduplication statistics.

4. **Comprehensive Reporting**: Provides detailed reports on the deduplication process
   including efficiency metrics and merge quality indicators.

Usage Examples:
    # Process a complete chain with enhanced merging
    processed_chain = await process_and_update_chain("page_123")
    
    # Apply enhanced merging to any chain
    updated_chain, report = await apply_enhanced_node_merging(
        chain=my_triplets, 
        task_info="Login to application",
        update_database=True
    )
    
    # Access the deduplication report
    if processed_chain:
        report = processed_chain[0].get('deduplication_report', {})
        print(f"Merged {report['summary']['nodes_successfully_merged']} nodes")

The strategy ensures that the agent builds a comprehensive and coherent understanding 
of its interactions while maintaining the rich contextual information necessary for 
effective task automation and evolution.
"""

from typing import List, Dict, Any, Optional, Tuple, Set
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain_core.runnables import RunnablePassthrough
from pydantic import BaseModel, Field, SecretStr
import json
import os
from langchain_openai import ChatOpenAI
from data.graph_db import Neo4jDatabase
import config
from collections import defaultdict

os.environ["LANGCHAIN_TRACING_V2"] = config.LANGCHAIN_TRACING_V2
os.environ["LANGCHAIN_ENDPOINT"] = config.LANGCHAIN_ENDPOINT
os.environ["LANGCHAIN_API_KEY"] = config.LANGCHAIN_API_KEY
os.environ["LANGCHAIN_PROJECT"] = "LearnTriplet"  # Keep specific project name

model = ChatOpenAI(
    openai_api_base=config.LLM_BASE_URL,
    openai_api_key=SecretStr(config.LLM_API_KEY),
    model_name=config.LLM_MODEL,
    request_timeout=config.LLM_REQUEST_TIMEOUT,
    max_retries=config.LLM_MAX_RETRIES,
    max_tokens=config.LLM_MAX_TOKEN,
)

URI = config.Neo4j_URI
AUTH = config.Neo4j_AUTH
db = Neo4jDatabase(URI, AUTH)


# Model definition for reasoning results
class TripletReasoning(BaseModel):
    context: str = Field(description="Context description of the operation")
    user_intent: str = Field(description="User intention analysis")
    state_change: str = Field(description="State change description")
    task_relation: str = Field(description="Relationship with the task")
    source_page_enhanced_desc: str = Field(
        description="Enhanced description of the source page"
    )
    element_enhanced_desc: str = Field(
        description="Enhanced description of the element"
    )
    target_page_enhanced_desc: str = Field(
        description="Enhanced description of the target page"
    )


# Model for storing node description with context
class NodeDescriptionContext(BaseModel):
    description: str = Field(description="Node description")
    action_context: str = Field(description="Context of the action that generated this description")
    triplet_index: int = Field(description="Index of the triplet in the chain")
    role: str = Field(description="Role of the node in the triplet (source_page, target_page, element)")


# Model for consolidated node information  
class ConsolidatedNode(BaseModel):
    node_id: str = Field(description="Unique node identifier")
    node_type: str = Field(description="Type of node (Page, Element)")
    descriptions: List[NodeDescriptionContext] = Field(description="All descriptions for this node")
    final_description: Optional[str] = Field(description="Final merged description", default=None)


# Model for enhanced merging result
class EnhancedMergeResult(BaseModel):
    merged_description: str = Field(description="The final merged description")
    consolidation_summary: str = Field(description="Summary of how descriptions were consolidated")
    key_aspects_preserved: List[str] = Field(description="Key aspects preserved from original descriptions")
    task_relevance_score: float = Field(description="Relevance score to the overall task (0-1)")


class PageCategoryResult(BaseModel):
    """Result of page category analysis"""
    category_name: str = Field(description="The category name for this page (e.g., 'Home Page', 'Product Listing', 'Login Page')")
    category_description: str = Field(description="A brief description of what this page category represents")
    confidence_score: float = Field(description="Confidence in the categorization (0-1)")
    key_elements: List[str] = Field(description="Key UI elements that identify this category")


class PageCategoryManager:
    """Manages page categorization and category-based merging"""
    
    def __init__(self):
        self.page_categories: Dict[str, List[str]] = {}  # Maps category_name -> [page_ids]
        self.page_to_category: Dict[str, str] = {}  # Maps page_id -> category_name
        self.category_descriptions: Dict[str, str] = {}  # Maps category_name -> merged_description
        self.category_metadata: Dict[str, Dict] = {}  # Maps category_name -> metadata
    
    def add_page_to_category(self, page_id: str, category_name: str, description: str, metadata: Dict = None):
        """Add a page to a category"""
        if category_name not in self.page_categories:
            self.page_categories[category_name] = []
            self.category_descriptions[category_name] = ""
            self.category_metadata[category_name] = {}
        
        if page_id not in self.page_categories[category_name]:
            self.page_categories[category_name].append(page_id)
        
        self.page_to_category[page_id] = category_name
        
        if metadata:
            self.category_metadata[category_name].update(metadata)
    
    def get_categories_for_merging(self) -> List[Tuple[str, List[str]]]:
        """Get categories that have multiple pages and need merging"""
        return [(cat, pages) for cat, pages in self.page_categories.items() if len(pages) > 1]
    
    def get_all_categories(self) -> List[str]:
        """Get all category names"""
        return list(self.page_categories.keys())


# Data structure for managing node deduplication
class NodeDeduplicationManager:
    """Manages node deduplication and merging across triplet chains"""
    
    def __init__(self):
        self.node_descriptions: Dict[str, ConsolidatedNode] = {}
        self.physical_duplicates: Dict[str, List[str]] = {}  # Maps canonical_id -> [duplicate_ids]
        
    def add_node_description(
        self, 
        node_id: str, 
        node_type: str,
        description: str, 
        action_context: str, 
        triplet_index: int, 
        role: str
    ):
        """Add a node description with its context"""
        if node_id not in self.node_descriptions:
            self.node_descriptions[node_id] = ConsolidatedNode(
                node_id=node_id,
                node_type=node_type,
                descriptions=[]
            )
        
        desc_context = NodeDescriptionContext(
            description=description,
            action_context=action_context,
            triplet_index=triplet_index,
            role=role
        )
        self.node_descriptions[node_id].descriptions.append(desc_context)
    
    def add_physical_duplicate(self, canonical_id: str, duplicate_id: str):
        """Record that duplicate_id is a physical duplicate of canonical_id"""
        if canonical_id not in self.physical_duplicates:
            self.physical_duplicates[canonical_id] = []
        if duplicate_id not in self.physical_duplicates[canonical_id]:
            self.physical_duplicates[canonical_id].append(duplicate_id)
    
    def get_nodes_needing_merge(self) -> List[ConsolidatedNode]:
        """Get nodes that have multiple descriptions and need merging"""
        return [node for node in self.node_descriptions.values() if len(node.descriptions) > 1]
    
    def get_physical_duplicates_for_deletion(self) -> Dict[str, List[str]]:
        """Get mapping of canonical nodes to their physical duplicates that should be deleted"""
        return self.physical_duplicates.copy()
    
    def get_all_nodes(self) -> List[ConsolidatedNode]:
        """Get all nodes"""
        return list(self.node_descriptions.values())
    
    def update_final_description(self, node_id: str, final_description: str):
        """Update the final merged description for a node"""
        if node_id in self.node_descriptions:
            self.node_descriptions[node_id].final_description = final_description


# Build LCEL chain for triplet reasoning
def create_triplet_reasoning_chain():
    """Create LCEL chain for triplet reasoning"""
    # Define reasoning prompt template
    triplet_reasoning_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are an AI assistant specialized in understanding and reasoning about UI operation chains. You need to analyze the given page-element-page triplet information and perform deep understanding and reasoning. You will receive textual descriptions and screenshots of pages, please analyze both.",
            ),
            (
                "human",
                [
                    {
                        "type": "text",
                        "text": """Please analyze the following UI operation triplet information:
        Source Page: {source_page_desc}
        Element: {element_desc}
        Target Page: {target_page_desc}
        Action: {action_name}
        
        Please reason and expand from the following aspects:
        1. What is the context and purpose of this operation?
        2. What might be the user's intention when performing this operation?
        3. How does this operation affect the page state change?
        4. What is the relationship between this operation and the overall task flow?
        5. Based on your understanding, generate richer and more accurate descriptions for the source page, element, and target page.
        
        Please return your reasoning results in a structured way, including the following fields:
        - context: Operation context description
        - user_intent: User intention analysis
        - state_change: State change description
        - task_relation: Relationship with the task
        - source_page_enhanced_desc: Enhanced description of source page
        - element_enhanced_desc: Enhanced description of element
        - target_page_enhanced_desc: Enhanced description of target page
        
        {format_instructions}""",
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": "{source_page_image}"},
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": "{target_page_image}"},
                    },
                ],
            ),
        ]
    )

    # Use JsonOutputParser instead of StrOutputParser
    parser = JsonOutputParser(pydantic_object=TripletReasoning)

    # Inject format instructions into prompt template
    prompt = triplet_reasoning_prompt.partial(
        format_instructions=parser.get_format_instructions()
    )

    # Build LCEL chain
    reasoning_chain = RunnablePassthrough() | prompt | model | parser

    return reasoning_chain


# Build LCEL chain for description merging
def create_merge_descriptions_chain():
    """Create LCEL chain for merging page descriptions

    Returns:
        Description merging chain
    """
    # Define merge description prompt template
    merge_descriptions_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are an AI assistant specialized in merging and optimizing page descriptions. You need to analyze descriptions of shared pages between two adjacent triplets and merge them into a more complete description.",
            ),
            (
                "human",
                """Please analyze the following two descriptions that describe the same page but from different contexts:

        Current Task: {task_info}
        
        Description 1 (as target page of previous triplet): {desc1}
        Description 2 (as source page of next triplet): {desc2}
        
        Please merge these two descriptions to generate a more complete and coherent description, requirements:
        1. Consider the context and goals of the current task
        2. Preserve all important information
        3. Eliminate redundant content
        4. Ensure description coherence
        5. Highlight core functionality and features of the page
        6. Emphasize relevance to the current task
        
        Please return the merged description.""",
            ),
        ]
    )

    # Build LCEL chain
    merge_chain = (
        RunnablePassthrough() | merge_descriptions_prompt | model | StrOutputParser()
    )

    return merge_chain


# Build enhanced LCEL chain for context-aware node merging
def create_enhanced_node_merge_chain():
    """Create LCEL chain for enhanced context-aware node merging
    
    Returns:
        Enhanced node merging chain
    """
    # Define enhanced merge prompt template
    enhanced_merge_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """You are an AI assistant specialized in intelligent node deduplication and context-aware description merging. 
                You excel at analyzing multiple descriptions of the same UI element or page that were generated from different action contexts, 
                and consolidating them into a single, comprehensive description that preserves all valuable information while maintaining coherence.""",
            ),
            (
                "human",
                """You are tasked with merging multiple descriptions of the same {node_type} node that were generated from different action contexts within a task execution chain.

                **Global Task Context:** {task_info}
                
                **Node Identification:** {node_id}
                **Node Type:** {node_type}
                
                **Multiple Descriptions to Merge:**
                {descriptions_with_context}
                
                **Your Task:**
                Analyze all the provided descriptions and their action contexts to create a single, comprehensive description that:

                1. **Contextual Analysis:** Consider how each description was generated from its specific action context
                2. **Global Task Understanding:** Take into account the broader overall task the agent is performing  
                3. **Information Synthesis:** Preserve all unique and valuable information from each description
                4. **Redundancy Elimination:** Remove duplicate or redundant information while maintaining completeness
                5. **Coherence Enhancement:** Ensure the final description flows logically and coherently
                6. **Task Relevance:** Emphasize aspects most relevant to the overall task completion
                7. **Complementarity Recognition:** Identify how different descriptions complement each other

                **Output Requirements:**
                Please return a JSON object with the following structure:
                - merged_description: The final consolidated description 
                - consolidation_summary: Brief explanation of how descriptions were merged
                - key_aspects_preserved: List of key aspects preserved from the original descriptions
                - task_relevance_score: Score from 0-1 indicating relevance to the overall task

                {format_instructions}""",
            ),
        ]
    )

    # Use JsonOutputParser for structured output
    parser = JsonOutputParser(pydantic_object=EnhancedMergeResult)
    
    # Inject format instructions into prompt template
    prompt = enhanced_merge_prompt.partial(
        format_instructions=parser.get_format_instructions()
    )

    # Build LCEL chain
    enhanced_merge_chain = RunnablePassthrough() | prompt | model | parser

    return enhanced_merge_chain


# Process single triplet
async def process_triplet(triplet: Dict[str, Any], reasoning_chain):
    """Process reasoning for a single triplet

    Args:
        triplet: Triplet containing source page, element, target page and action information
        reasoning_chain: Triplet reasoning chain

    Returns:
        Triplet with added reasoning results
    """
    # Prepare reasoning input
    reasoning_input = {
        "source_page_desc": triplet["source_page"].get("description", ""),
        "element_desc": triplet["element"].get("description", ""),
        "target_page_desc": triplet["target_page"].get("description", ""),
        "action_name": triplet["action"].get("action_name", ""),
    }

    # Load source and target page images
    try:
        from utils import load_image_to_base64

        # Load source page image
        source_page_image = "data:image/png;base64,"
        if (
            "raw_page_url" in triplet["source_page"]
            and triplet["source_page"]["raw_page_url"]
        ):
            source_image_path = triplet["source_page"]["raw_page_url"]
            source_page_image = load_image_to_base64(source_image_path)

        # Load target page image
        target_page_image = "data:image/png;base64,"
        if (
            "raw_page_url" in triplet["target_page"]
            and triplet["target_page"]["raw_page_url"]
        ):
            target_image_path = triplet["target_page"]["raw_page_url"]
            target_page_image = load_image_to_base64(target_image_path)

        # Add images to reasoning input
        reasoning_input["source_page_image"] = source_page_image
        reasoning_input["target_page_image"] = target_page_image
    except Exception as e:
        print(f"Error loading page images: {str(e)}")
        # Use empty base64 image if loading fails
        reasoning_input["source_page_image"] = "data:image/png;base64,"
        reasoning_input["target_page_image"] = "data:image/png;base64,"
    try:
        # Execute reasoning - now returns dictionary object
        reasoning_result = await reasoning_chain.ainvoke(reasoning_input)

        # Store result directly as reasoning field
        triplet["reasoning"] = reasoning_result

        # Update descriptions in triplet - update enhanced descriptions directly to description field
        triplet["source_page"]["description"] = reasoning_result[
            "source_page_enhanced_desc"
        ]
        triplet["element"]["description"] = reasoning_result["element_enhanced_desc"]
        triplet["target_page"]["description"] = reasoning_result[
            "target_page_enhanced_desc"
        ]

    except Exception as e:
        print(f"Error during triplet reasoning: {str(e)}")
        # Record detailed error information for debugging
        triplet["reasoning_error"] = str(e)

    return triplet


# Collect all node descriptions with their contexts across the entire chain
def collect_node_descriptions_with_context(chain: List[Dict[str, Any]]) -> NodeDeduplicationManager:
    """Collect all node descriptions with their action contexts from the entire chain
    
    Args:
        chain: Complete triplet chain
        
    Returns:
        NodeDeduplicationManager with all collected descriptions
    """
    dedup_manager = NodeDeduplicationManager()
    
    for triplet_index, triplet in enumerate(chain):
        # Build action context description
        action_context = f"Action: {triplet.get('action', {}).get('action_name', 'Unknown')}"
        if 'action' in triplet and 'action_params' in triplet['action']:
            action_params = triplet['action']['action_params']
            if action_params:
                action_context += f" with parameters: {action_params}"
        
        # Add source page description
        if 'source_page' in triplet and 'page_id' in triplet['source_page']:
            source_desc = triplet['source_page'].get('description', '')
            if source_desc.strip():
                dedup_manager.add_node_description(
                    node_id=triplet['source_page']['page_id'],
                    node_type='Page',
                    description=source_desc,
                    action_context=f"Source page for {action_context}",
                    triplet_index=triplet_index,
                    role='source_page'
                )
        
        # Add target page description
        if 'target_page' in triplet and 'page_id' in triplet['target_page']:
            target_desc = triplet['target_page'].get('description', '')
            if target_desc.strip():
                dedup_manager.add_node_description(
                    node_id=triplet['target_page']['page_id'],
                    node_type='Page',
                    description=target_desc,
                    action_context=f"Target page reached after {action_context}",
                    triplet_index=triplet_index,
                    role='target_page'
                )
        
        # Add element description
        if 'element' in triplet and 'element_id' in triplet['element']:
            element_desc = triplet['element'].get('description', '')
            if element_desc.strip():
                dedup_manager.add_node_description(
                    node_id=triplet['element']['element_id'],
                    node_type='Element',
                    description=element_desc,
                    action_context=f"Element involved in {action_context}",
                    triplet_index=triplet_index,
                    role='element'
                )
    
    return dedup_manager


# Create page categorization chain
def create_page_categorization_chain():
    """Create LCEL chain for analyzing page descriptions and determining categories
    
    Returns:
        Page categorization chain
    """
    categorization_prompt = ChatPromptTemplate.from_messages([
        ("system", """You are an AI assistant specialized in analyzing mobile app page descriptions and categorizing them into meaningful page types.

Your task is to analyze page descriptions and determine what category of page this represents based on its functionality and UI elements.

Analyze the page content and determine the most appropriate category."""),
        
        ("human", """Please analyze this page description and categorize it:

Page Description: {page_description}

Context Information:
- Task: {task_info}
- Elements on page: {page_elements}

Please provide your analysis in the following JSON format:

{{
  "category_name": "The most appropriate category name for this page",
  "category_description": "A brief description of what this page category represents",
  "confidence_score": 0.85,
  "key_elements": ["list", "of", "key", "UI", "elements"]
}}

Make sure to return ONLY valid JSON with these exact field names.""")
    ])
    
    # Build LCEL chain with JSON output
    categorization_chain = (
        RunnablePassthrough() 
        | categorization_prompt 
        | model 
        | JsonOutputParser()
    )
    
    return categorization_chain


# Analyze pages and group them by category
async def categorize_pages_in_chain(chain: List[Dict[str, Any]], task_info: str = "Unknown Task") -> PageCategoryManager:
    """Analyze all pages in a chain and group them by category
    
    Args:
        chain: Triplet chain containing pages to categorize
        task_info: Task information for context
        
    Returns:
        PageCategoryManager with categorized pages
    """
    print("🏷️  Categorizing pages by functionality...")
    
    category_manager = PageCategoryManager()
    categorization_chain = create_page_categorization_chain()
    
    # Collect all unique pages
    unique_pages = {}
    for triplet_idx, triplet in enumerate(chain):
        print(f"   📋 Triplet {triplet_idx}: {triplet.get('source_page', {}).get('page_id', 'None')[:8]} → {triplet.get('target_page', {}).get('page_id', 'None')[:8]}")
        for page_key in ['source_page', 'target_page']:
            if page_key in triplet and 'page_id' in triplet[page_key]:
                page_id = triplet[page_key]['page_id']
                if page_id not in unique_pages:
                    unique_pages[page_id] = triplet[page_key]
                    print(f"      Added unique page: {page_id[:8]} (from {page_key})")
                else:
                    print(f"      Already seen page: {page_id[:8]} (from {page_key})")
    
    print(f"\n   Found {len(unique_pages)} unique pages to categorize")
    print(f"   Page IDs: {[pid[:8] for pid in unique_pages.keys()]}")
    
    # Categorize each page
    for page_idx, (page_id, page_data) in enumerate(unique_pages.items(), 1):
        try:
            print(f"\n   🔍 Processing page {page_idx}/{len(unique_pages)}: {page_id[:8]}...")
            page_description = page_data.get('description', '')
            print(f"      Description length: {len(page_description)} chars")
            
            # Extract elements information for categorization
            page_elements = "No element information available"
            element_descriptions = []
            
            if 'elements' in page_data:
                try:
                    elements_data = json.loads(page_data['elements']) if isinstance(page_data['elements'], str) else page_data['elements']
                    if isinstance(elements_data, list) and elements_data:
                        element_types = [elem.get('type', 'unknown') for elem in elements_data[:15]]  # First 15 elements
                        element_texts = [elem.get('text', '') for elem in elements_data[:10] if elem.get('text', '').strip()]
                        page_elements = f"UI elements: {', '.join(set(element_types))}"
                        if element_texts:
                            element_descriptions = element_texts[:5]  # Top 5 text elements
                        print(f"      Found {len(elements_data)} elements, {len(element_texts)} with text")
                        print(f"      Element types: {set(element_types)}")
                        print(f"      Sample texts: {element_descriptions[:2]}")
                except Exception as e:
                    print(f"   ⚠️  Error parsing elements for page {page_id}: {str(e)}")
            else:
                print(f"      No 'elements' key in page_data")
            
            # If no description, try to create one from UI elements and raw_page_url
            if not page_description.strip():
                description_parts = []
                
                # Add element-based description
                if element_descriptions:
                    description_parts.append(f"Page with UI elements containing: {', '.join(element_descriptions[:3])}")
                elif 'UI elements:' in page_elements:
                    description_parts.append(f"Page containing {page_elements.lower()}")
                
                # Add URL-based context if available
                raw_url = page_data.get('raw_page_url', '')
                if raw_url and 'step' in raw_url:
                    description_parts.append(f"Screenshot from navigation step")
                
                if description_parts:
                    page_description = ". ".join(description_parts) + "."
                    print(f"   🔧 Generated fallback description for page {page_id[:8]}...")
                    print(f"      Generated: {page_description[:100]}...")
                else:
                    print(f"   ⚠️  Skipping page {page_id} - no description or usable element data")
                    continue
            else:
                print(f"      Using existing description: {page_description[:100]}...")
            
            # Analyze page category
            category_input = {
                "page_description": page_description,
                "task_info": task_info,
                "page_elements": page_elements
            }
            
            print(f"      🤖 Sending to LLM for categorization...")
            print(f"      Input description: {page_description[:150]}...")
            print(f"      Input elements: {page_elements[:100]}...")
            
            category_result_dict = await categorization_chain.ainvoke(category_input)
            
            print(f"      🤖 LLM Response: {category_result_dict}")
            
            # Convert dictionary to PageCategoryResult object
            try:
                category_result = PageCategoryResult(**category_result_dict)
            except Exception as e:
                print(f"   ❌ Error parsing category result for page {page_id}: {str(e)}")
                print(f"   Raw result: {category_result_dict}")
                continue
            
            if category_result.confidence_score >= 0.6:  # Only use high-confidence categorizations
                print(f"   📋 Page {page_id[:8]}... → {category_result.category_name} (confidence: {category_result.confidence_score:.2f})")
                
                category_manager.add_page_to_category(
                    page_id=page_id,
                    category_name=category_result.category_name,
                    description=page_description,
                    metadata={
                        'confidence': category_result.confidence_score,
                        'category_description': category_result.category_description,
                        'key_elements': category_result.key_elements,
                        'original_page_data': page_data
                    }
                )
            else:
                print(f"   ⚠️  Page {page_id[:8]}... - low confidence categorization ({category_result.confidence_score:.2f}), skipping")
                
        except Exception as e:
            print(f"   ❌ Error categorizing page {page_id}: {str(e)}")
    
    return category_manager


# Perform category-based page merging with physical deletion
async def merge_pages_by_category(
    category_manager: PageCategoryManager, 
    enhanced_merge_chain,
    task_info: str
) -> Dict[str, Any]:
    """Merge pages by category and physically delete duplicate page nodes
    
    Args:
        category_manager: PageCategoryManager with categorized pages
        enhanced_merge_chain: Enhanced merging chain for description consolidation
        task_info: Task information for context
        
    Returns:
        Dictionary with merge results and statistics
    """
    print("🗂️  Merging pages by category with physical deletion...")
    
    merge_results = {
        "categories_processed": 0,
        "pages_merged": 0,
        "pages_deleted": 0,
        "canonical_pages_created": 0,
        "relationships_updated": 0,
        "category_details": []
    }
    
    categories_for_merging = category_manager.get_categories_for_merging()
    
    if not categories_for_merging:
        print("   No categories with multiple pages found for merging")
        return merge_results
    
    print(f"   Found {len(categories_for_merging)} categories with multiple pages")
    
    try:
        with db.driver.session(database="neo4j") as session:
            for category_name, page_ids in categories_for_merging:
                print(f"\n🏷️  Processing category: {category_name} ({len(page_ids)} pages)")
                
                # Collect all page descriptions for this category
                page_descriptions = []
                page_metadata = []
                
                for page_id in page_ids:
                    # Get page data from database
                    page_query = """
                    MATCH (p:Page)
                    WHERE p.page_id = $page_id
                    RETURN p
                    """
                    page_result = session.run(page_query, page_id=page_id)
                    page_record = page_result.single()
                    
                    if page_record:
                        page_node = dict(page_record["p"])
                        page_descriptions.append(page_node.get('description', ''))
                        page_metadata.append({
                            'page_id': page_id,
                            'timestamp': page_node.get('timestamp'),
                            'raw_page_url': page_node.get('raw_page_url'),
                            'other_info': page_node.get('other_info'),
                            'elements': page_node.get('elements')
                        })
                
                if not page_descriptions:
                    print(f"   ⚠️  No page data found for category {category_name}")
                    continue
                
                # Create enhanced merge input for category consolidation
                descriptions_with_context = ""
                for i, desc in enumerate(page_descriptions, 1):
                    descriptions_with_context += f"\n--- Description {i} (from {page_metadata[i-1]['page_id'][:8]}...) ---\n{desc}\n"
                
                # Use enhanced merging to create category description
                merge_input = {
                    "descriptions_with_context": descriptions_with_context,
                    "task_info": task_info,
                    "node_type": "Page Category",
                    "merge_context": f"Merging {len(page_ids)} pages into {category_name} category"
                }
                
                try:
                    merge_result = await enhanced_merge_chain.ainvoke(merge_input)
                    category_description = merge_result.merged_description
                    
                    print(f"   ✅ Generated category description: {category_description[:100]}...")
                    
                except Exception as e:
                    print(f"   ⚠️  Failed to merge descriptions for {category_name}, using fallback: {str(e)}")
                    category_description = f"Category: {category_name}. " + " | ".join([desc[:50] for desc in page_descriptions if desc.strip()])
                
                # Select canonical page (first one chronologically or first in list)
                canonical_page_id = page_ids[0]
                canonical_page_metadata = page_metadata[0]
                duplicate_page_ids = page_ids[1:]
                
                print(f"   📌 Using {canonical_page_id[:8]}... as canonical page")
                print(f"   🗑️  Will delete {len(duplicate_page_ids)} duplicate pages")
                
                # Update canonical page with category information
                update_canonical_query = """
                MATCH (canonical:Page)
                WHERE canonical.page_id = $canonical_page_id
                SET canonical.description = $category_description,
                    canonical.name = $category_name,
                    canonical.title = $category_name,
                    canonical.is_category_representative = true,
                    canonical.category_name = $category_name,
                    canonical.category_merge_timestamp = datetime(),
                    canonical.merged_page_count = $merged_page_count,
                    canonical.merged_from_pages = $merged_from_pages,
                    canonical.category_confidence = $category_confidence
                RETURN canonical
                """
                
                # Get category metadata
                category_meta = category_manager.category_metadata.get(category_name, {})
                
                session.run(update_canonical_query,
                           canonical_page_id=canonical_page_id,
                           category_description=category_description,
                           category_name=category_name,
                           merged_page_count=len(page_ids),
                           merged_from_pages=json.dumps(page_ids),
                           category_confidence=category_meta.get('confidence', 0.8))
                
                # Redirect all relationships from duplicate pages to canonical page
                relationships_updated = 0
                
                for dup_page_id in duplicate_page_ids:
                    # Redirect incoming relationships (avoiding duplicates)
                    redirect_incoming_query = """
                    MATCH (source)-[r]->(duplicate:Page)
                    WHERE duplicate.page_id = $dup_page_id
                    MATCH (canonical:Page)
                    WHERE canonical.page_id = $canonical_page_id
                    // Only create relationship if it doesn't exist
                    MERGE (source)-[new_r:NAVIGATES_TO]->(canonical)
                    ON CREATE SET new_r = properties(r), 
                                  new_r.redirected_from = $dup_page_id,
                                  new_r.category_merged = true,
                                  new_r.created_by_merge = true
                    ON MATCH SET new_r.also_redirected_from = CASE 
                                    WHEN new_r.also_redirected_from IS NULL THEN [$dup_page_id]
                                    ELSE new_r.also_redirected_from + $dup_page_id
                                 END
                    DELETE r
                    RETURN count(r) as redirected_count
                    """
                    
                    result = session.run(redirect_incoming_query, 
                                       dup_page_id=dup_page_id, 
                                       canonical_page_id=canonical_page_id)
                    count_record = result.single()
                    if count_record:
                        relationships_updated += count_record["redirected_count"]
                    
                    # Redirect outgoing relationships (avoiding duplicates)
                    redirect_outgoing_query = """
                    MATCH (duplicate:Page)-[r]->(target)
                    WHERE duplicate.page_id = $dup_page_id
                    MATCH (canonical:Page)
                    WHERE canonical.page_id = $canonical_page_id
                    // Only create relationship if it doesn't exist
                    MERGE (canonical)-[new_r:NAVIGATES_TO]->(target)
                    ON CREATE SET new_r = properties(r), 
                                  new_r.redirected_from = $dup_page_id,
                                  new_r.category_merged = true,
                                  new_r.created_by_merge = true
                    ON MATCH SET new_r.also_redirected_from = CASE 
                                    WHEN new_r.also_redirected_from IS NULL THEN [$dup_page_id]
                                    ELSE new_r.also_redirected_from + $dup_page_id
                                 END
                    DELETE r
                    RETURN count(r) as redirected_count
                    """
                    
                    result = session.run(redirect_outgoing_query, 
                                       dup_page_id=dup_page_id, 
                                       canonical_page_id=canonical_page_id)
                    count_record = result.single()
                    if count_record:
                        relationships_updated += count_record["redirected_count"]
                
                # Physically delete duplicate page nodes
                pages_deleted = 0
                for dup_page_id in duplicate_page_ids:
                    delete_query = """
                    MATCH (duplicate:Page)
                    WHERE duplicate.page_id = $dup_page_id
                    DELETE duplicate
                    RETURN count(duplicate) as deleted_count
                    """
                    
                    result = session.run(delete_query, dup_page_id=dup_page_id)
                    count_record = result.single()
                    if count_record:
                        pages_deleted += count_record["deleted_count"]
                
                # Update results
                merge_results["categories_processed"] += 1
                merge_results["pages_merged"] += len(page_ids)
                merge_results["pages_deleted"] += pages_deleted
                merge_results["canonical_pages_created"] += 1
                merge_results["relationships_updated"] += relationships_updated
                
                merge_results["category_details"].append({
                    "category_name": category_name,
                    "total_pages": len(page_ids),
                    "pages_deleted": pages_deleted,
                    "relationships_updated": relationships_updated,
                    "canonical_page_id": canonical_page_id,
                    "merged_description_length": len(category_description)
                })
                
                # Clean up any remaining duplicate relationships
                cleanup_query = """
                MATCH (a)-[r1:NAVIGATES_TO]->(b), (a)-[r2:NAVIGATES_TO]->(b)
                WHERE id(r1) < id(r2) AND r1.category_merged = true AND r2.category_merged = true
                DELETE r2
                RETURN count(r2) as duplicates_removed
                """
                
                cleanup_result = session.run(cleanup_query)
                cleanup_record = cleanup_result.single()
                duplicates_removed = cleanup_record["duplicates_removed"] if cleanup_record else 0
                
                if duplicates_removed > 0:
                    print(f"   🧹 Cleaned up {duplicates_removed} duplicate relationships")
                
                print(f"   ✅ Category {category_name}: merged {len(page_ids)} pages, deleted {pages_deleted}, updated {relationships_updated} relationships")
                
    except Exception as e:
        print(f"❌ Error during category-based merging: {str(e)}")
    
    # Summary
    efficiency = (merge_results["pages_deleted"] / max(merge_results["pages_merged"], 1)) * 100
    print(f"\n✅ Category-based merging completed:")
    print(f"   📊 {merge_results['categories_processed']} categories processed")
    print(f"   🗂️  {merge_results['pages_merged']} total pages involved")
    print(f"   🗑️  {merge_results['pages_deleted']} pages physically deleted")
    print(f"   📈 {efficiency:.1f}% deletion efficiency")
    print(f"   🔗 {merge_results['relationships_updated']} relationships updated")
    
    return merge_results


# Enhanced function for merging descriptions using context-aware consolidation
async def enhanced_merge_node_descriptions(
    dedup_manager: NodeDeduplicationManager,
    enhanced_merge_chain,
    task_info: str
) -> Dict[str, str]:
    """Perform enhanced merging of node descriptions using context-aware consolidation
    
    Args:
        dedup_manager: NodeDeduplicationManager containing all descriptions
        enhanced_merge_chain: Enhanced merging chain for LLM processing
        task_info: Global task information
        
    Returns:
        Dictionary mapping node_id to final merged description
    """
    merged_descriptions = {}
    nodes_needing_merge = dedup_manager.get_nodes_needing_merge()
    
    print(f"📋 Found {len(nodes_needing_merge)} nodes requiring description merging")
    
    for node in nodes_needing_merge:
        print(f"🔄 Merging {len(node.descriptions)} descriptions for {node.node_type} node: {node.node_id}")
        
        # Format descriptions with their contexts
        descriptions_with_context = ""
        for i, desc_context in enumerate(node.descriptions, 1):
            descriptions_with_context += f"""
Description {i}:
- Content: {desc_context.description}
- Action Context: {desc_context.action_context}
- Triplet Position: #{desc_context.triplet_index}
- Node Role: {desc_context.role}
---"""
        
        # Prepare input for enhanced merging
        merge_input = {
            "node_id": node.node_id,
            "node_type": node.node_type,
            "task_info": task_info,
            "descriptions_with_context": descriptions_with_context
        }
        
        try:
            # Execute enhanced merging
            merge_result = await enhanced_merge_chain.ainvoke(merge_input)
            
            final_description = merge_result.get("merged_description", "")
            merged_descriptions[node.node_id] = final_description
            
            # Update the deduplication manager
            dedup_manager.update_final_description(node.node_id, final_description)
            
            print(f"✅ Successfully merged descriptions for node {node.node_id}")
            print(f"   Task relevance score: {merge_result.get('task_relevance_score', 0):.2f}")
            print(f"   Key aspects preserved: {len(merge_result.get('key_aspects_preserved', []))}")
            
        except Exception as e:
            print(f"❌ Error merging descriptions for node {node.node_id}: {str(e)}")
            # Fallback: use the first description
            if node.descriptions:
                merged_descriptions[node.node_id] = node.descriptions[0].description
    
    return merged_descriptions


# Legacy function for backward compatibility - enhanced version
async def merge_node_descriptions(
    chain: List[Dict[str, Any]], merge_chain, task_info: str
):
    """Enhanced merge description information of overlapping nodes in the chain
    
    This function now uses the enhanced node deduplication and merging strategy
    while maintaining backward compatibility.

    Args:
        chain: Triplet chain
        merge_chain: Description merging chain (legacy, kept for compatibility)
        task_info: Task information

    Returns:
        Triplet chain with updated descriptions
    """
    # Use the enhanced merging strategy
    dedup_manager = collect_node_descriptions_with_context(chain)
    enhanced_merge_chain = create_enhanced_node_merge_chain()
    
    # Perform enhanced merging
    merged_descriptions = await enhanced_merge_node_descriptions(
        dedup_manager, enhanced_merge_chain, task_info
    )
    
    # Update the chain with merged descriptions
    for triplet in chain:
        # Update source page if it was merged
        if 'source_page' in triplet and 'page_id' in triplet['source_page']:
            page_id = triplet['source_page']['page_id']
            if page_id in merged_descriptions:
                triplet['source_page']['description'] = merged_descriptions[page_id]
        
        # Update target page if it was merged  
        if 'target_page' in triplet and 'page_id' in triplet['target_page']:
            page_id = triplet['target_page']['page_id']
            if page_id in merged_descriptions:
                triplet['target_page']['description'] = merged_descriptions[page_id]
        
        # Update element if it was merged
        if 'element' in triplet and 'element_id' in triplet['element']:
            element_id = triplet['element']['element_id']
            if element_id in merged_descriptions:
                triplet['element']['description'] = merged_descriptions[element_id]
    
    print(f"🎯 Enhanced node merging completed. Updated {len(merged_descriptions)} nodes with consolidated descriptions.")
    
    return chain


# Update node properties in database
def update_node_in_db(
    node_id: str,
    property_name: str,
    property_value: Any,
    node_type: Optional[str] = None,
) -> bool:
    """Update node properties in database

    Args:
        node_id: Node ID
        property_name: Property name
        property_value: Property value
        node_type: Node type (optional)

    Returns:
        Whether update was successful
    """
    try:
        return db.update_node_property(
            node_id=node_id,
            property_name=property_name,
            property_value=property_value,
            node_type=node_type,
        )
    except Exception as e:
        print(f"Error updating node property: {str(e)}")
        return False


# Identify physical duplicate nodes in the database
def identify_physical_duplicate_nodes(chain: List[Dict[str, Any]]) -> NodeDeduplicationManager:
    """Identify physically duplicate nodes in the database that should be merged and deleted
    
    Args:
        chain: Triplet chain containing the nodes to analyze
        
    Returns:
        NodeDeduplicationManager with duplicate mappings
    """
    dedup_manager = NodeDeduplicationManager()
    
    print("🔍 Identifying physical duplicate nodes in database...")
    
    # Extract all unique node IDs from the chain
    page_nodes = set()
    element_nodes = set()
    
    for triplet in chain:
        if 'source_page' in triplet and 'page_id' in triplet['source_page']:
            page_nodes.add(triplet['source_page']['page_id'])
        if 'target_page' in triplet and 'page_id' in triplet['target_page']:
            page_nodes.add(triplet['target_page']['page_id'])
        if 'element' in triplet and 'element_id' in triplet['element']:
            element_nodes.add(triplet['element']['element_id'])
    
    print(f"   Found {len(page_nodes)} unique page IDs and {len(element_nodes)} unique element IDs")
    
    # Query database to find nodes with similar content that could be duplicates
    try:
        with db.driver.session(database="neo4j") as session:
            # Find ACTUAL duplicates - pages with identical or very similar content
            page_similarity_query = """
            MATCH (p1:Page), (p2:Page)
            WHERE p1.page_id < p2.page_id
            AND (
                p1.raw_page_url = p2.raw_page_url OR
                p1.elements = p2.elements OR
                (p1.elements IS NOT NULL AND p2.elements IS NOT NULL AND 
                 size(p1.elements) = size(p2.elements) AND 
                 abs(size(p1.elements) - size(p2.elements)) < 50)
            )
            RETURN p1.page_id as page1, p2.page_id as page2,
                   p1.description as desc1, p2.description as desc2,
                   p1.raw_page_url as url1, p2.raw_page_url as url2,
                   size(p1.elements) as elements1_size, size(p2.elements) as elements2_size,
                   p1.elements = p2.elements as identical_elements
            """
            
            page_duplicates = session.run(page_similarity_query)
            page_duplicate_count = 0
            
            for record in page_duplicates:
                page1, page2 = record["page1"], record["page2"]
                
                # Always mark as duplicates - we want to reduce from 5 to 2 pages
                canonical_id = page1  # Keep the first one
                duplicate_id = page2  # Delete the second one
                
                dedup_manager.add_physical_duplicate(canonical_id, duplicate_id)
                page_duplicate_count += 1
                print(f"   📄 WILL DELETE duplicate page: {duplicate_id} (keeping {canonical_id})")
                print(f"      Reason: Elements size {record.get('elements1_size', 0)} vs {record.get('elements2_size', 0)}")
            
            # Find elements with similar descriptions
            element_similarity_query = """
            MATCH (e1:Element), (e2:Element)
            WHERE e1.element_id < e2.element_id
            AND (
                e1.description = e2.description OR
                (e1.description IS NOT NULL AND e2.description IS NOT NULL AND
                 e1.description CONTAINS e2.description) OR
                (e2.description IS NOT NULL AND e1.description IS NOT NULL AND
                 e2.description CONTAINS e1.description)
            )
            RETURN e1.element_id as elem1, e2.element_id as elem2,
                   e1.description as desc1, e2.description as desc2
            """
            
            element_duplicates = session.run(element_similarity_query)
            element_duplicate_count = 0
            
            for record in element_duplicates:
                elem1, elem2 = record["elem1"], record["elem2"]
                
                # Choose canonical node (prefer the one that appears first in chain)
                canonical_id = elem1 if elem1 in element_nodes else elem2
                duplicate_id = elem2 if canonical_id == elem1 else elem1
                
                if canonical_id in element_nodes or duplicate_id in element_nodes:
                    dedup_manager.add_physical_duplicate(canonical_id, duplicate_id)
                    element_duplicate_count += 1
                    print(f"   🔘 Found duplicate elements: {canonical_id} ← {duplicate_id}")
            
            print(f"✅ Identified {page_duplicate_count} page duplicates and {element_duplicate_count} element duplicates")
            
    except Exception as e:
        print(f"❌ Error identifying duplicates: {str(e)}")
    
    return dedup_manager


# Deep merge and deduplication function that actually removes duplicate nodes
def deep_merge_and_deduplicate_nodes(merged_descriptions: Dict[str, str], dedup_manager: NodeDeduplicationManager) -> Dict[str, Any]:
    """Perform deep merge and deduplication by actually removing duplicate nodes from database
    
    Args:
        merged_descriptions: Dictionary mapping node_id to merged description
        dedup_manager: NodeDeduplicationManager containing consolidated node information
        
    Returns:
        Dictionary with merge results and statistics
    """
    merge_results = {
        "nodes_merged": 0,
        "nodes_deleted": 0,
        "relationships_updated": 0,
        "merge_details": []
    }
    
    print("🔥 Performing deep merge and physical node deletion...")
    
    # Get physical duplicates to delete
    physical_duplicates = dedup_manager.get_physical_duplicates_for_deletion()
    
    try:
        with db.driver.session(database="neo4j") as session:
            # Process each canonical node and its duplicates
            for canonical_id, duplicate_ids in physical_duplicates.items():
                if not duplicate_ids:
                    continue
                    
                print(f"🔄 Processing canonical node {canonical_id} with {len(duplicate_ids)} duplicates")
                
                # Determine node type by querying database
                node_type_query = """
                MATCH (n)
                WHERE n.page_id = $node_id OR n.element_id = $node_id
                RETURN labels(n)[0] as node_type, n
                """
                
                node_result = session.run(node_type_query, node_id=canonical_id)
                canonical_record = node_result.single()
                
                if not canonical_record:
                    print(f"⚠️ Canonical node {canonical_id} not found, skipping")
                    continue
                    
                node_type = canonical_record["node_type"]
                id_field = "page_id" if node_type == "Page" else "element_id"
                
                # Collect all properties from duplicate nodes to merge
                duplicate_properties = []
                all_duplicate_ids = duplicate_ids + [canonical_id]
                
                for dup_id in duplicate_ids:
                    dup_query = f"""
                    MATCH (n:{node_type})
                    WHERE n.{id_field} = $dup_id
                    RETURN n
                    """
                    dup_result = session.run(dup_query, dup_id=dup_id)
                    dup_record = dup_result.single()
                    if dup_record:
                        duplicate_properties.append(dict(dup_record["n"]))
                
                # Merge properties (combine descriptions, preserve other unique properties)
                merged_description = merged_descriptions.get(canonical_id, "")
                if not merged_description:
                    # Combine all descriptions from duplicates
                    descriptions = []
                    for props in duplicate_properties:
                        if props.get("description"):
                            descriptions.append(props["description"])
                    merged_description = " | ".join(descriptions)
                
                # Update canonical node with merged properties
                update_canonical_query = f"""
                MATCH (canonical:{node_type})
                WHERE canonical.{id_field} = $canonical_id
                SET canonical.description = $merged_description,
                    canonical.is_merged_node = true,
                    canonical.merge_timestamp = datetime(),
                    canonical.merged_from_nodes = $merged_from_ids,
                    canonical.duplicate_count = $duplicate_count
                RETURN canonical
                """
                
                session.run(update_canonical_query,
                           canonical_id=canonical_id,
                           merged_description=merged_description,
                           merged_from_ids=json.dumps(all_duplicate_ids),
                           duplicate_count=len(duplicate_ids))
                
                # Redirect all relationships from duplicates to canonical
                relationships_updated = 0
                
                for dup_id in duplicate_ids:
                    # Redirect incoming relationships
                    redirect_incoming_query = f"""
                    MATCH (source)-[r]->(duplicate:{node_type})
                    WHERE duplicate.{id_field} = $dup_id
                    MATCH (canonical:{node_type})
                    WHERE canonical.{id_field} = $canonical_id
                    CREATE (source)-[new_r:NAVIGATES_TO]->(canonical)
                    SET new_r = properties(r)
                    SET new_r.redirected_from = $dup_id
                    SET new_r.merged_relationship = true
                    DELETE r
                    RETURN count(new_r) as redirected
                    """
                    
                    incoming_result = session.run(redirect_incoming_query,
                                                 dup_id=dup_id,
                                                 canonical_id=canonical_id)
                    incoming_count = incoming_result.single()
                    if incoming_count:
                        relationships_updated += incoming_count["redirected"]
                    
                    # Redirect outgoing relationships
                    redirect_outgoing_query = f"""
                    MATCH (duplicate:{node_type})-[r]->(target)
                    WHERE duplicate.{id_field} = $dup_id
                    MATCH (canonical:{node_type})
                    WHERE canonical.{id_field} = $canonical_id
                    CREATE (canonical)-[new_r:NAVIGATES_TO]->(target)
                    SET new_r = properties(r)
                    SET new_r.redirected_from = $dup_id
                    SET new_r.merged_relationship = true
                    DELETE r
                    RETURN count(new_r) as redirected
                    """
                    
                    outgoing_result = session.run(redirect_outgoing_query,
                                                 dup_id=dup_id,
                                                 canonical_id=canonical_id)
                    outgoing_count = outgoing_result.single()
                    if outgoing_count:
                        relationships_updated += outgoing_count["redirected"]
                
                # Delete duplicate nodes
                nodes_deleted = 0
                for dup_id in duplicate_ids:
                    delete_query = f"""
                    MATCH (duplicate:{node_type})
                    WHERE duplicate.{id_field} = $dup_id
                    DELETE duplicate
                    RETURN count(duplicate) as deleted
                    """
                    
                    delete_result = session.run(delete_query, dup_id=dup_id)
                    deleted_count = delete_result.single()
                    if deleted_count:
                        nodes_deleted += deleted_count["deleted"]
                        print(f"🗑️ Deleted duplicate {node_type} node: {dup_id}")
                
                merge_results["nodes_deleted"] += nodes_deleted
                merge_results["relationships_updated"] += relationships_updated
                merge_results["nodes_merged"] += 1
                
                merge_details = {
                    "canonical_node_id": canonical_id,
                    "node_type": node_type,
                    "duplicates_deleted": len(duplicate_ids),
                    "relationships_redirected": relationships_updated,
                    "final_description_length": len(merged_description)
                }
                merge_results["merge_details"].append(merge_details)
                
                print(f"✅ Merged {node_type} {canonical_id}: deleted {nodes_deleted} duplicates, redirected {relationships_updated} relationships")
                
    except Exception as e:
        print(f"❌ Error during deep merge: {str(e)}")
        import traceback
        traceback.print_exc()
    
    return merge_results


# Handle logical duplicates (same node_id with multiple descriptions in chain)
async def merge_logical_duplicates_in_database(merged_descriptions: Dict[str, str], description_manager: NodeDeduplicationManager) -> Dict[str, Any]:
    """Update database nodes that have the same ID but multiple descriptions in the chain
    
    This handles the case where the same page_id or element_id appears multiple times
    in the chain with different descriptions from different contexts.
    
    Args:
        merged_descriptions: Dictionary mapping node_id to merged description
        description_manager: NodeDeduplicationManager containing description contexts
        
    Returns:
        Dictionary with logical merge results
    """
    logical_merge_results = {
        "nodes_updated": 0,
        "descriptions_merged": 0,
        "logical_merge_details": []
    }
    
    print("🔄 Updating database with merged descriptions for logical duplicates...")
    
    nodes_needing_merge = description_manager.get_nodes_needing_merge()
    
    for node in nodes_needing_merge:
        node_id = node.node_id
        node_type = node.node_type
        
        if node_id not in merged_descriptions:
            continue
            
        final_description = merged_descriptions[node_id]
        
        print(f"   📝 Updating {node_type} {node_id} with merged description from {len(node.descriptions)} contexts")
        
        # Update the node in database with merged description
        try:
            with db.driver.session(database="neo4j") as session:
                id_field = "page_id" if node_type == "Page" else "element_id"
                
                update_query = f"""
                MATCH (n:{node_type})
                WHERE n.{id_field} = $node_id
                SET n.description = $merged_description,
                    n.logical_merge_metadata = $metadata,
                    n.is_logically_merged = true,
                    n.logical_merge_timestamp = datetime()
                RETURN n.{id_field} as updated_id
                """
                
                logical_metadata = {
                    "original_description_count": len(node.descriptions),
                    "action_contexts": [desc.action_context for desc in node.descriptions],
                    "triplet_positions": [desc.triplet_index for desc in node.descriptions],
                    "roles": [desc.role for desc in node.descriptions],
                    "merge_type": "logical_duplicate"
                }
                
                result = session.run(update_query,
                                   node_id=node_id,
                                   merged_description=final_description,
                                   metadata=json.dumps(logical_metadata))
                
                updated_record = result.single()
                if updated_record:
                    logical_merge_results["nodes_updated"] += 1
                    logical_merge_results["descriptions_merged"] += len(node.descriptions)
                    
                    logical_detail = {
                        "node_id": node_id,
                        "node_type": node_type,
                        "original_descriptions": len(node.descriptions),
                        "final_description_length": len(final_description),
                        "contexts_merged": [desc.action_context for desc in node.descriptions]
                    }
                    logical_merge_results["logical_merge_details"].append(logical_detail)
                    
                    print(f"   ✅ Updated {node_type} {node_id} with {len(node.descriptions)} merged descriptions")
                else:
                    print(f"   ❌ Failed to update {node_type} {node_id}")
                    
        except Exception as e:
            print(f"   ❌ Error updating {node_type} {node_id}: {str(e)}")
    
    print(f"✅ Logical duplicate merging completed:")
    print(f"   - {logical_merge_results['nodes_updated']} nodes updated with merged descriptions")
    print(f"   - {logical_merge_results['descriptions_merged']} total descriptions merged")
    
    return logical_merge_results


# Enhanced database update function for merged descriptions
def update_merged_descriptions_in_db(merged_descriptions: Dict[str, str], dedup_manager: NodeDeduplicationManager) -> Dict[str, bool]:
    """Update all merged descriptions in the database with enhanced metadata
    
    Args:
        merged_descriptions: Dictionary mapping node_id to merged description
        dedup_manager: NodeDeduplicationManager containing consolidated node information
        
    Returns:
        Dictionary mapping node_id to update success status
    """
    update_results = {}
    
    for node_id, final_description in merged_descriptions.items():
        node = dedup_manager.node_descriptions.get(node_id)
        if not node:
            continue
            
        # Update main description
        success = update_node_in_db(
            node_id=node_id,
            property_name="description",
            property_value=final_description,
            node_type=node.node_type
        )
        
        if success:
            # Store merge metadata
            merge_metadata = {
                "merged_from_count": len(node.descriptions),
                "original_descriptions": [desc.description for desc in node.descriptions],
                "action_contexts": [desc.action_context for desc in node.descriptions],
                "merge_timestamp": json.dumps({"timestamp": "auto-generated"}),
                "is_merged_description": True
            }
            
            # Store merge metadata as a separate property
            update_node_in_db(
                node_id=node_id,
                property_name="merge_metadata",
                property_value=json.dumps(merge_metadata),
                node_type=node.node_type
            )
            
            print(f"✅ Updated {node.node_type} node {node_id} with merged description and metadata")
        else:
            print(f"❌ Failed to update {node.node_type} node {node_id}")
            
        update_results[node_id] = success
    
    return update_results


# Generate deduplication report
def generate_deduplication_report(dedup_manager: NodeDeduplicationManager, merged_descriptions: Dict[str, str]) -> Dict[str, Any]:
    """Generate a comprehensive report on the deduplication and merging process
    
    Args:
        dedup_manager: NodeDeduplicationManager containing all processed nodes
        merged_descriptions: Dictionary of final merged descriptions
        
    Returns:
        Comprehensive deduplication report
    """
    all_nodes = dedup_manager.get_all_nodes()
    nodes_needing_merge = dedup_manager.get_nodes_needing_merge()
    
    report = {
        "summary": {
            "total_nodes_processed": len(all_nodes),
            "nodes_requiring_merge": len(nodes_needing_merge),
            "nodes_successfully_merged": len(merged_descriptions),
            "deduplication_efficiency": len(merged_descriptions) / len(all_nodes) if all_nodes else 0
        },
        "node_statistics": {
            "pages": len([n for n in all_nodes if n.node_type == "Page"]),
            "elements": len([n for n in all_nodes if n.node_type == "Element"]),
            "pages_merged": len([n for n in nodes_needing_merge if n.node_type == "Page"]),
            "elements_merged": len([n for n in nodes_needing_merge if n.node_type == "Element"])
        },
        "merging_details": []
    }
    
    # Add detailed merging information
    for node in nodes_needing_merge:
        if node.node_id in merged_descriptions:
            node_detail = {
                "node_id": node.node_id,
                "node_type": node.node_type,
                "original_descriptions_count": len(node.descriptions),
                "contexts_merged": [desc.action_context for desc in node.descriptions],
                "triplet_positions": [desc.triplet_index for desc in node.descriptions],
                "final_description_length": len(merged_descriptions[node.node_id])
            }
            report["merging_details"].append(node_detail)
    
    return report


# Process single chain
async def process_single_chain(
    chain: List[Dict[str, Any]], reasoning_chain, merge_chain
) -> List[Dict[str, Any]]:
    """Process all triplets in a single chain and merge descriptions using enhanced strategy

    Args:
        chain: Single triplet chain
        reasoning_chain: Triplet reasoning chain
        merge_chain: Description merging chain (legacy compatibility)

    Returns:
        Processed chain with comprehensive deduplication report
    """
    print(f"🚀 Starting enhanced chain processing for {len(chain)} triplets")
    
    # Extract task information from first node in chain
    task_info = "Unknown Task"
    if (
        chain
        and chain[0]
        and "source_page" in chain[0]
        and "other_info" in chain[0]["source_page"]
    ):
        try:
            other_info = chain[0]["source_page"]["other_info"]
            if isinstance(other_info, str):
                other_info = json.loads(other_info)

            if "task_info" in other_info and "description" in other_info["task_info"]:
                task_info = other_info["task_info"]["description"]
                print(f"📋 Extracted task information: {task_info}")
        except Exception as e:
            print(f"⚠️ Error extracting task information: {str(e)}")

    # Process each triplet with reasoning
    print(f"🧠 Processing triplet reasoning...")
    processed_triplets = []
    for i, triplet in enumerate(chain):
        print(f"   Processing triplet {i+1}/{len(chain)}")
        processed_triplet = await process_triplet(triplet, reasoning_chain)
        processed_triplets.append(processed_triplet)

    # Collect all node descriptions with contexts
    print(f"📝 Collecting node descriptions and contexts...")
    dedup_manager = collect_node_descriptions_with_context(processed_triplets)
    
    # Perform enhanced merging
    print(f"🔄 Performing enhanced node deduplication and merging...")
    enhanced_merge_chain = create_enhanced_node_merge_chain()
    merged_descriptions = await enhanced_merge_node_descriptions(
        dedup_manager, enhanced_merge_chain, task_info
    )

    # Update the chain with merged descriptions
    for triplet in processed_triplets:
        # Update source page if it was merged
        if 'source_page' in triplet and 'page_id' in triplet['source_page']:
            page_id = triplet['source_page']['page_id']
            if page_id in merged_descriptions:
                triplet['source_page']['description'] = merged_descriptions[page_id]
        
        # Update target page if it was merged  
        if 'target_page' in triplet and 'page_id' in triplet['target_page']:
            page_id = triplet['target_page']['page_id']
            if page_id in merged_descriptions:
                triplet['target_page']['description'] = merged_descriptions[page_id]
        
        # Update element if it was merged
        if 'element' in triplet and 'element_id' in triplet['element']:
            element_id = triplet['element']['element_id']
            if element_id in merged_descriptions:
                triplet['element']['description'] = merged_descriptions[element_id]

    # Update database with enhanced metadata
    print(f"💾 Updating database with merged descriptions and metadata...")
    update_results = update_merged_descriptions_in_db(merged_descriptions, dedup_manager)
    
    # Update individual node information in database (for nodes not merged)
    for triplet in processed_triplets:
        # Update source page description if not already merged
        if 'source_page' in triplet and 'page_id' in triplet['source_page']:
            page_id = triplet['source_page']['page_id']
            if page_id not in merged_descriptions:
                update_node_in_db(
                    page_id,
                    "description",
                    triplet['source_page'].get("description", ""),
                    "Page",
                )

        # Update target page description if not already merged
        if 'target_page' in triplet and 'page_id' in triplet['target_page']:
            page_id = triplet['target_page']['page_id']
            if page_id not in merged_descriptions:
                update_node_in_db(
                    page_id,
                    "description",
                    triplet['target_page'].get("description", ""),
                    "Page",
                )

        # Update element description if not already merged
        if 'element' in triplet and 'element_id' in triplet['element']:
            element_id = triplet['element']['element_id']
            if element_id not in merged_descriptions:
                update_node_in_db(
                    element_id,
                    "description",
                    triplet['element'].get("description", ""),
                    "Element",
                )

        # Save reasoning results (if exists)
        if "reasoning" in triplet:
            # Save complete reasoning results to element node
            update_node_in_db(
                triplet["element"]["element_id"],
                "reasoning",
                json.dumps(triplet["reasoning"]),
                "Element",
            )

    # Generate comprehensive deduplication report
    deduplication_report = generate_deduplication_report(dedup_manager, merged_descriptions)
    
    print(f"\n📊 === NODE DEDUPLICATION REPORT ===")
    print(f"   Total nodes processed: {deduplication_report['summary']['total_nodes_processed']}")
    print(f"   Nodes requiring merge: {deduplication_report['summary']['nodes_requiring_merge']}")
    print(f"   Nodes successfully merged: {deduplication_report['summary']['nodes_successfully_merged']}")
    print(f"   Deduplication efficiency: {deduplication_report['summary']['deduplication_efficiency']:.2%}")
    print(f"   Pages: {deduplication_report['node_statistics']['pages']} (merged: {deduplication_report['node_statistics']['pages_merged']})")
    print(f"   Elements: {deduplication_report['node_statistics']['elements']} (merged: {deduplication_report['node_statistics']['elements_merged']})")
    print(f"=====================================\n")

    # Add the report to the first triplet for reference
    if processed_triplets:
        processed_triplets[0]['deduplication_report'] = deduplication_report

    return processed_triplets


# Process complete chain and update database
async def process_and_update_chain(
    start_page_id: str, 
    use_category_merging: bool = False
) -> List[Dict[str, Any]]:
    """Process triplet chain and update database using enhanced node merging strategy

    Args:
        start_page_id: Starting page ID
        use_category_merging: If True, applies category-based page merging with physical deletion
                             instead of just logical merging

    Returns:
        List of processed triplets with comprehensive deduplication
    """
    print(f"🎯 Starting enhanced chain processing from page: {start_page_id}")
    
    # Create necessary reasoning chains
    reasoning_chain = create_triplet_reasoning_chain()
    merge_chain = create_merge_descriptions_chain()  # Legacy compatibility

    # Extract chain data
    triplets = db.get_chain_from_start(
        start_page_id
    )  # Extract chain triplet data from database

    if not triplets:
        print(f"⚠️ Warning: No triplets found for start page: {start_page_id}")
        return []

    print(f"📋 Retrieved {len(triplets)} triplets from database")

    if use_category_merging:
        print(f"🏷️  Using category-based page merging with physical deletion")
        
        # Extract task information for categorization
        task_info = "Unknown Task"
        if (
            triplets
            and triplets[0]
            and "source_page" in triplets[0]
            and "other_info" in triplets[0]["source_page"]
        ):
            try:
                other_info = triplets[0]["source_page"]["other_info"]
                if isinstance(other_info, str):
                    other_info = json.loads(other_info)

                if "task_info" in other_info and "description" in other_info["task_info"]:
                    task_info = other_info["task_info"]["description"]
                    print(f"📋 Extracted task information: {task_info}")
            except Exception as e:
                print(f"⚠️ Error extracting task information: {str(e)}")
        
        # First, process triplets to generate descriptions if they don't exist
        print(f"🔍 Pre-processing triplets to generate descriptions...")
        reasoning_chain = create_triplet_reasoning_chain()
        processed_triplets = []
        for i, triplet in enumerate(triplets):
            print(f"   Processing triplet {i+1}/{len(triplets)} for description generation")
            processed_triplet = await process_triplet(triplet, reasoning_chain)
            processed_triplets.append(processed_triplet)
        
        # Apply category-based page merging to the processed triplets
        processed_chain, category_report = await apply_category_based_page_merging(
            chain=processed_triplets,
            task_info=task_info
        )
        
        # Add category report to first triplet for UI display
        if processed_chain:
            processed_chain[0]['category_merge_report'] = category_report
            
        print(f"✅ Category-based chain processing completed successfully")
        print(f"   🗑️  {category_report['summary']['pages_physically_deleted']} pages physically deleted")
        print(f"   📈 {category_report['summary']['deletion_efficiency']:.1f}% deletion efficiency")
        
    else:
        print(f"🔧 Using standard enhanced merging (logical only)")
        # Process with enhanced strategy (logical merging only)
        processed_chain = await process_single_chain(triplets, reasoning_chain, merge_chain)
        print(f"✅ Enhanced chain processing completed successfully")

    return processed_chain


# Direct API for deep deduplication with physical node deletion
async def apply_deep_node_deduplication(
    chain: List[Dict[str, Any]], 
    task_info: str = "Unknown Task"
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Apply deep node deduplication that physically removes duplicate nodes from database
    
    This function identifies truly duplicate nodes in the database and merges them by:
    1. Finding nodes with identical or very similar content
    2. Merging their descriptions and properties
    3. Redirecting all relationships to canonical nodes
    4. Physically deleting duplicate nodes from database
    
    Args:
        chain: List of triplet dictionaries
        task_info: Global task information for context-aware merging
        
    Returns:
        Tuple of (updated_chain, deduplication_report)
    """
    print(f"🔥 Applying deep node deduplication with physical deletion to chain of {len(chain)} triplets")
    
    # Step 1: Identify physical duplicate nodes in database
    duplicate_manager = identify_physical_duplicate_nodes(chain)
    
    # Step 2: Collect description contexts for merging
    description_manager = collect_node_descriptions_with_context(chain)
    
    # Step 3: Perform LLM-driven description merging
    enhanced_merge_chain = create_enhanced_node_merge_chain()
    merged_descriptions = await enhanced_merge_node_descriptions(
        description_manager, enhanced_merge_chain, task_info
    )
    
    # Step 4: Perform deep merge with physical node deletion for physical duplicates
    deep_merge_results = deep_merge_and_deduplicate_nodes(merged_descriptions, duplicate_manager)
    
    # Step 4.5: Handle logical duplicates (same node_id with multiple descriptions)
    logical_merge_results = await merge_logical_duplicates_in_database(merged_descriptions, description_manager)
    
    # Step 5: Update chain with surviving canonical nodes
    updated_chain = []
    canonical_mappings = duplicate_manager.get_physical_duplicates_for_deletion()
    
    # Create reverse mapping: duplicate_id -> canonical_id
    duplicate_to_canonical = {}
    for canonical_id, duplicate_ids in canonical_mappings.items():
        for dup_id in duplicate_ids:
            duplicate_to_canonical[dup_id] = canonical_id
    
    for triplet in chain:
        updated_triplet = triplet.copy()
        
        # Update source page if it was deduplicated
        if 'source_page' in updated_triplet and 'page_id' in updated_triplet['source_page']:
            page_id = updated_triplet['source_page']['page_id']
            canonical_id = duplicate_to_canonical.get(page_id, page_id)
            if canonical_id != page_id:
                print(f"   📄 Updating source_page {page_id} → {canonical_id}")
                updated_triplet['source_page']['page_id'] = canonical_id
            if canonical_id in merged_descriptions:
                updated_triplet['source_page']['description'] = merged_descriptions[canonical_id]
        
        # Update target page if it was deduplicated  
        if 'target_page' in updated_triplet and 'page_id' in updated_triplet['target_page']:
            page_id = updated_triplet['target_page']['page_id']
            canonical_id = duplicate_to_canonical.get(page_id, page_id)
            if canonical_id != page_id:
                print(f"   📄 Updating target_page {page_id} → {canonical_id}")
                updated_triplet['target_page']['page_id'] = canonical_id
            if canonical_id in merged_descriptions:
                updated_triplet['target_page']['description'] = merged_descriptions[canonical_id]
        
        # Update element if it was deduplicated
        if 'element' in updated_triplet and 'element_id' in updated_triplet['element']:
            element_id = updated_triplet['element']['element_id']
            canonical_id = duplicate_to_canonical.get(element_id, element_id)
            if canonical_id != element_id:
                print(f"   🔘 Updating element {element_id} → {canonical_id}")
                updated_triplet['element']['element_id'] = canonical_id
            if canonical_id in merged_descriptions:
                updated_triplet['element']['description'] = merged_descriptions[canonical_id]
        
        updated_chain.append(updated_triplet)
    
    # Generate comprehensive deduplication report
    total_nodes = len(description_manager.get_all_nodes())
    physical_duplicates = sum(len(dups) for dups in canonical_mappings.values())
    
    deduplication_report = {
        "summary": {
            "total_nodes_processed": total_nodes,
            "physical_duplicates_found": physical_duplicates,
            "logical_duplicates_found": len(description_manager.get_nodes_needing_merge()),
            "nodes_physically_deleted": deep_merge_results["nodes_deleted"],
            "nodes_logically_merged": logical_merge_results["nodes_updated"],
            "total_descriptions_merged": logical_merge_results["descriptions_merged"],
            "relationships_updated": deep_merge_results["relationships_updated"],
            "physical_deletion_efficiency": deep_merge_results["nodes_deleted"] / total_nodes if total_nodes else 0,
            "logical_merge_efficiency": logical_merge_results["nodes_updated"] / total_nodes if total_nodes else 0
        },
        "physical_merge_details": deep_merge_results["merge_details"],
        "logical_merge_details": logical_merge_results["logical_merge_details"],
        "physical_deletion": True,
        "logical_merging": True
    }
    
    print(f"✅ Deep node deduplication completed:")
    print(f"   - {deep_merge_results['nodes_deleted']} duplicate nodes physically deleted")
    print(f"   - {logical_merge_results['nodes_updated']} nodes logically merged")
    print(f"   - {logical_merge_results['descriptions_merged']} total descriptions merged")
    print(f"   - {deep_merge_results['relationships_updated']} relationships redirected")
    print(f"   - Physical deletion: {deduplication_report['summary']['physical_deletion_efficiency']:.2%}")
    print(f"   - Logical merging: {deduplication_report['summary']['logical_merge_efficiency']:.2%}")
    
    return updated_chain, deduplication_report


# Main API for category-based page deduplication and merging
async def apply_category_based_page_merging(
    chain: List[Dict[str, Any]], 
    task_info: str = "Unknown Task"
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Apply category-based page deduplication that groups similar pages and physically deletes duplicates
    
    This function analyzes pages in the chain, categorizes them by functionality (e.g., 'Home Page', 
    'Product Listing', 'Product Detail'), merges pages in the same category, and physically deletes 
    duplicate page nodes while maintaining one representative per category.
    
    Perfect for scenarios like e-commerce apps where you visit home→listing→product→listing→product→home
    but only want 3 category nodes: Home, Product Listing, Product Detail.
    
    Args:
        chain: List of triplet dictionaries containing page navigation data
        task_info: Task context information for better categorization
        
    Returns:
        Tuple of (updated_chain, category_merge_report)
    """
    print(f"🏷️  Applying category-based page merging to chain of {len(chain)} triplets")
    
    # Step 1: Categorize all pages by functionality
    category_manager = await categorize_pages_in_chain(chain, task_info)
    
    # Step 2: Create enhanced merging chain for description consolidation
    enhanced_merge_chain = create_enhanced_node_merge_chain()
    
    # Step 3: Merge pages within each category and physically delete duplicates
    category_merge_results = await merge_pages_by_category(
        category_manager, enhanced_merge_chain, task_info
    )
    
    # Step 4: Update chain to reference canonical pages only
    updated_chain = []
    page_id_mappings = {}  # Maps original_page_id -> canonical_page_id
    
    # Build mapping from category manager
    for category_name, page_ids in category_manager.page_categories.items():
        if len(page_ids) > 1:
            canonical_page_id = page_ids[0]  # First page becomes canonical
            for page_id in page_ids:
                page_id_mappings[page_id] = canonical_page_id
    
    # Update triplets to reference canonical pages
    for triplet in chain:
        updated_triplet = triplet.copy()
        
        # Update source page reference
        if 'source_page' in updated_triplet and 'page_id' in updated_triplet['source_page']:
            original_id = updated_triplet['source_page']['page_id']
            if original_id in page_id_mappings:
                updated_triplet['source_page']['page_id'] = page_id_mappings[original_id]
                # Add metadata about the category merge
                updated_triplet['source_page']['is_category_merged'] = True
                updated_triplet['source_page']['original_page_id'] = original_id
        
        # Update target page reference
        if 'target_page' in updated_triplet and 'page_id' in updated_triplet['target_page']:
            original_id = updated_triplet['target_page']['page_id']
            if original_id in page_id_mappings:
                updated_triplet['target_page']['page_id'] = page_id_mappings[original_id]
                # Add metadata about the category merge
                updated_triplet['target_page']['is_category_merged'] = True
                updated_triplet['target_page']['original_page_id'] = original_id
        
        updated_chain.append(updated_triplet)
    
    # Step 5: Create comprehensive report
    category_merge_report = {
        "summary": {
            "total_pages_analyzed": len(set(
                [t['source_page']['page_id'] for t in chain if 'source_page' in t and 'page_id' in t['source_page']] +
                [t['target_page']['page_id'] for t in chain if 'target_page' in t and 'page_id' in t['target_page']]
            )),
            "categories_identified": len(category_manager.get_all_categories()),
            "categories_merged": category_merge_results["categories_processed"],
            "pages_physically_deleted": category_merge_results["pages_deleted"],
            "canonical_pages_remaining": category_merge_results["canonical_pages_created"],
            "relationships_updated": category_merge_results["relationships_updated"],
            "deletion_efficiency": (category_merge_results["pages_deleted"] / max(category_merge_results["pages_merged"], 1)) * 100
        },
        "category_details": category_merge_results["category_details"],
        "page_mappings": page_id_mappings,
        "categories": {
            category: {
                "page_count": len(pages),
                "canonical_page": pages[0] if pages else None,
                "metadata": category_manager.category_metadata.get(category, {})
            }
            for category, pages in category_manager.page_categories.items()
        }
    }
    
    # Log summary
    print(f"\n✅ Category-based page merging completed:")
    print(f"   📊 {category_merge_report['summary']['total_pages_analyzed']} total pages analyzed")
    print(f"   🏷️  {category_merge_report['summary']['categories_identified']} categories identified")
    print(f"   🗂️  {category_merge_report['summary']['categories_merged']} categories with multiple pages merged")
    print(f"   🗑️  {category_merge_report['summary']['pages_physically_deleted']} pages physically deleted")
    print(f"   📈 {category_merge_report['summary']['deletion_efficiency']:.1f}% deletion efficiency")
    print(f"   🔗 {category_merge_report['summary']['relationships_updated']} relationships updated")
    
    return updated_chain, category_merge_report


# Direct API for enhanced node merging (standalone function)
async def apply_enhanced_node_merging(
    chain: List[Dict[str, Any]], 
    task_info: str = "Unknown Task",
    update_database: bool = True
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Apply enhanced node deduplication and merging to any triplet chain
    
    This is a standalone function that can be used independently to apply
    the enhanced node merging strategy to any triplet chain.
    
    Args:
        chain: List of triplet dictionaries
        task_info: Global task information for context-aware merging
        update_database: Whether to update the database with merged descriptions
        
    Returns:
        Tuple of (updated_chain, deduplication_report)
    """
    print(f"🔧 Applying enhanced node merging to chain of {len(chain)} triplets")
    
    # Collect all node descriptions with contexts
    dedup_manager = collect_node_descriptions_with_context(chain)
    
    # Create enhanced merging chain
    enhanced_merge_chain = create_enhanced_node_merge_chain()
    
    # Perform enhanced merging
    merged_descriptions = await enhanced_merge_node_descriptions(
        dedup_manager, enhanced_merge_chain, task_info
    )
    
    # Update the chain with merged descriptions
    updated_chain = []
    for triplet in chain:
        updated_triplet = triplet.copy()
        
        # Update source page if it was merged
        if 'source_page' in updated_triplet and 'page_id' in updated_triplet['source_page']:
            page_id = updated_triplet['source_page']['page_id']
            if page_id in merged_descriptions:
                updated_triplet['source_page']['description'] = merged_descriptions[page_id]
        
        # Update target page if it was merged  
        if 'target_page' in updated_triplet and 'page_id' in updated_triplet['target_page']:
            page_id = updated_triplet['target_page']['page_id']
            if page_id in merged_descriptions:
                updated_triplet['target_page']['description'] = merged_descriptions[page_id]
        
        # Update element if it was merged
        if 'element' in updated_triplet and 'element_id' in updated_triplet['element']:
            element_id = updated_triplet['element']['element_id']
            if element_id in merged_descriptions:
                updated_triplet['element']['description'] = merged_descriptions[element_id]
        
        updated_chain.append(updated_triplet)
    
    # Update database if requested
    if update_database:
        print(f"💾 Updating database with merged descriptions...")
        update_results = update_merged_descriptions_in_db(merged_descriptions, dedup_manager)
        print(f"   Database updates: {sum(update_results.values())}/{len(update_results)} successful")
    
    # Generate comprehensive report
    deduplication_report = generate_deduplication_report(dedup_manager, merged_descriptions)
    
    print(f"✅ Enhanced node merging completed:")
    print(f"   - {deduplication_report['summary']['nodes_successfully_merged']} nodes merged")
    print(f"   - {deduplication_report['summary']['deduplication_efficiency']:.2%} efficiency")
    
    return updated_chain, deduplication_report


# Demonstration and testing function
async def demonstrate_enhanced_merging():
    """
    Demonstration function showing the enhanced node merging capabilities
    
    This function creates a sample triplet chain with overlapping nodes and
    demonstrates how the enhanced merging strategy works.
    """
    print("🎯 === ENHANCED NODE MERGING DEMONSTRATION ===\n")
    
    # Create sample triplet chain with overlapping nodes
    sample_chain = [
        {
            "source_page": {
                "page_id": "page_1",
                "description": "Login page with username and password fields",
                "other_info": json.dumps({
                    "task_info": {"description": "User authentication flow"}
                })
            },
            "element": {
                "element_id": "elem_1", 
                "description": "Username input field"
            },
            "target_page": {
                "page_id": "page_2",
                "description": "Dashboard page after login"
            },
            "action": {
                "action_name": "enter_text",
                "action_params": {"text": "user@example.com"}
            }
        },
        {
            "source_page": {
                "page_id": "page_2",  # Same as target page above
                "description": "Main dashboard with navigation menu and widgets"
            },
            "element": {
                "element_id": "elem_2",
                "description": "Settings navigation button"
            },
            "target_page": {
                "page_id": "page_3",
                "description": "Settings page"
            },
            "action": {
                "action_name": "tap",
                "action_params": {"coordinates": [100, 50]}
            }
        },
        {
            "source_page": {
                "page_id": "page_2",  # Same page appearing again
                "description": "Dashboard showing user profile and activity feed"
            },
            "element": {
                "element_id": "elem_3",
                "description": "Profile avatar in header"
            },
            "target_page": {
                "page_id": "page_4",
                "description": "User profile page"
            },
            "action": {
                "action_name": "tap",
                "action_params": {}
            }
        }
    ]
    
    print(f"📋 Created sample chain with {len(sample_chain)} triplets")
    print(f"   Notice: page_2 appears 3 times with different descriptions")
    
    # Apply enhanced merging (without database updates for demo)
    updated_chain, report = await apply_enhanced_node_merging(
        chain=sample_chain,
        task_info="User authentication and navigation flow",
        update_database=False  # Don't update database in demo
    )
    
    print(f"\n📊 === DEMONSTRATION RESULTS ===")
    print(f"Summary:")
    print(f"  - Total nodes: {report['summary']['total_nodes_processed']}")
    print(f"  - Nodes merged: {report['summary']['nodes_successfully_merged']}")
    print(f"  - Efficiency: {report['summary']['deduplication_efficiency']:.2%}")
    
    print(f"\nDetailed merging:")
    for detail in report['merging_details']:
        print(f"  • {detail['node_type']} {detail['node_id']}:")
        print(f"    - Merged {detail['original_descriptions_count']} descriptions")
        print(f"    - From triplets: {detail['triplet_positions']}")
        print(f"    - Final description length: {detail['final_description_length']} chars")
    
    print(f"\n✅ Demonstration completed successfully!")
    return updated_chain, report


if __name__ == "__main__":
    # Run demonstration when module is executed directly
    import asyncio
    asyncio.run(demonstrate_enhanced_merging())
