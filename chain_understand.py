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
            
            # Find elements with similar descriptions or properties
            element_similarity_query = """
            MATCH (e1:Element), (e2:Element)
            WHERE e1.element_id < e2.element_id
            AND (
                e1.description = e2.description OR
                (e1.element_type = e2.element_type AND 
                 e1.description IS NOT NULL AND e2.description IS NOT NULL AND
                 e1.description CONTAINS e2.description) OR
                (e1.element_type = e2.element_type AND 
                 e2.description IS NOT NULL AND e1.description IS NOT NULL AND
                 e2.description CONTAINS e1.description)
            )
            RETURN e1.element_id as elem1, e2.element_id as elem2,
                   e1.description as desc1, e2.description as desc2,
                   e1.element_type as type1, e2.element_type as type2
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
                    CREATE (source)-[new_r:{{type(r)}}]->(canonical)
                    SET new_r = properties(r)
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
                    CREATE (canonical)-[new_r:{{type(r)}}]->(target)
                    SET new_r = properties(r)
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
async def process_and_update_chain(start_page_id: str) -> List[Dict[str, Any]]:
    """Process triplet chain and update database using enhanced node merging strategy

    Args:
        start_page_id: Starting page ID

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

    # Process with enhanced strategy
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
