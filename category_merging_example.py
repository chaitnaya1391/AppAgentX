#!/usr/bin/env python3
"""
Category-Based Page Merging Example

This example demonstrates how to use the new category-based page merging functionality
to consolidate pages by their functional categories instead of keeping duplicate page visits.

Perfect for scenarios like e-commerce apps where you visit:
home → listing → product → listing → product → home

Result: 3 category nodes (Home, Product Listing, Product Detail) instead of 6+ individual pages.
"""

import asyncio
from chain_understand import apply_category_based_page_merging

# Example triplet chain representing e-commerce navigation
example_ecommerce_chain = [
    {
        "source_page": {
            "page_id": "home_001",
            "description": "Main homepage with featured products, search bar, and navigation menu. Shows recommended items and promotional banners.",
            "elements": '[{"type": "search_bar"}, {"type": "navigation_menu"}, {"type": "product_card"}, {"type": "banner"}]'
        },
        "action": {"action_name": "tap", "action_params": {"element": "search_button"}},
        "target_page": {
            "page_id": "listing_001", 
            "description": "Product listing page showing search results with filters, sorting options, and product grid layout.",
            "elements": '[{"type": "filter_panel"}, {"type": "sort_dropdown"}, {"type": "product_grid"}, {"type": "pagination"}]'
        },
        "element": {"element_id": "search_btn_01", "description": "Search button on homepage"}
    },
    {
        "source_page": {
            "page_id": "listing_001",
            "description": "Product listing page showing search results with filters, sorting options, and product grid layout.",
            "elements": '[{"type": "filter_panel"}, {"type": "sort_dropdown"}, {"type": "product_grid"}, {"type": "pagination"}]'
        },
        "action": {"action_name": "tap", "action_params": {"element": "product_item"}},
        "target_page": {
            "page_id": "product_001",
            "description": "Product detail page with images, description, reviews, add to cart button, and related products.",
            "elements": '[{"type": "image_gallery"}, {"type": "description_text"}, {"type": "add_to_cart"}, {"type": "reviews_section"}]'
        },
        "element": {"element_id": "product_item_01", "description": "Product item in listing"}
    },
    {
        "source_page": {
            "page_id": "product_001",
            "description": "Product detail page with images, description, reviews, add to cart button, and related products.",
            "elements": '[{"type": "image_gallery"}, {"type": "description_text"}, {"type": "add_to_cart"}, {"type": "reviews_section"}]'
        },
        "action": {"action_name": "tap", "action_params": {"element": "back_button"}},
        "target_page": {
            "page_id": "listing_002",  # Same listing page, different visit
            "description": "Product listing page with search results, filters, and product grid. User returned from product detail.",
            "elements": '[{"type": "filter_panel"}, {"type": "sort_dropdown"}, {"type": "product_grid"}, {"type": "pagination"}]'
        },
        "element": {"element_id": "back_btn_01", "description": "Back button on product page"}
    },
    {
        "source_page": {
            "page_id": "listing_002",
            "description": "Product listing page with search results, filters, and product grid. User returned from product detail.",
            "elements": '[{"type": "filter_panel"}, {"type": "sort_dropdown"}, {"type": "product_grid"}, {"type": "pagination"}]'
        },
        "action": {"action_name": "tap", "action_params": {"element": "different_product"}},
        "target_page": {
            "page_id": "product_002",  # Different product, same type of page
            "description": "Product detail page displaying different item with images, reviews, specifications, and purchase options.",
            "elements": '[{"type": "image_gallery"}, {"type": "description_text"}, {"type": "add_to_cart"}, {"type": "reviews_section"}]'
        },
        "element": {"element_id": "product_item_02", "description": "Different product item in listing"}
    },
    {
        "source_page": {
            "page_id": "product_002",
            "description": "Product detail page displaying different item with images, reviews, specifications, and purchase options.",
            "elements": '[{"type": "image_gallery"}, {"type": "description_text"}, {"type": "add_to_cart"}, {"type": "reviews_section"}]'
        },
        "action": {"action_name": "tap", "action_params": {"element": "home_button"}},
        "target_page": {
            "page_id": "home_002",  # Same homepage, different visit
            "description": "Homepage displaying featured products, search functionality, and main navigation. User returned from product browsing.",
            "elements": '[{"type": "search_bar"}, {"type": "navigation_menu"}, {"type": "product_card"}, {"type": "banner"}]'
        },
        "element": {"element_id": "home_btn_01", "description": "Home button"}
    }
]


async def demonstrate_category_merging():
    """Demonstrate category-based page merging"""
    
    print("🛍️  E-commerce Category Merging Demonstration")
    print("=" * 60)
    
    # Show original chain structure
    print("\n📊 Original Chain Structure:")
    unique_pages = set()
    for triplet in example_ecommerce_chain:
        if 'source_page' in triplet:
            unique_pages.add(triplet['source_page']['page_id'])
        if 'target_page' in triplet:
            unique_pages.add(triplet['target_page']['page_id'])
    
    print(f"   Total triplets: {len(example_ecommerce_chain)}")
    print(f"   Unique page IDs: {len(unique_pages)}")
    print(f"   Page IDs: {', '.join(sorted(unique_pages))}")
    
    # Apply category-based merging
    print(f"\n🏷️  Applying category-based page merging...")
    updated_chain, merge_report = await apply_category_based_page_merging(
        chain=example_ecommerce_chain,
        task_info="E-commerce product browsing and shopping"
    )
    
    # Show results
    print(f"\n📈 Merging Results:")
    print(f"   Categories identified: {merge_report['summary']['categories_identified']}")
    print(f"   Pages physically deleted: {merge_report['summary']['pages_physically_deleted']}")
    print(f"   Deletion efficiency: {merge_report['summary']['deletion_efficiency']:.1f}%")
    
    print(f"\n🏷️  Categories Found:")
    for category, details in merge_report['categories'].items():
        print(f"   📋 {category}: {details['page_count']} pages → 1 canonical page")
        if 'metadata' in details and 'category_description' in details['metadata']:
            print(f"      Description: {details['metadata']['category_description']}")
    
    print(f"\n🔄 Page ID Mappings:")
    for original, canonical in merge_report['page_mappings'].items():
        print(f"   {original} → {canonical}")
    
    print(f"\n📊 Final Graph Structure:")
    final_unique_pages = set()
    for triplet in updated_chain:
        if 'source_page' in triplet:
            final_unique_pages.add(triplet['source_page']['page_id'])
        if 'target_page' in triplet:
            final_unique_pages.add(triplet['target_page']['page_id'])
    
    print(f"   Unique pages after merging: {len(final_unique_pages)}")
    print(f"   Canonical page IDs: {', '.join(sorted(final_unique_pages))}")
    
    print(f"\n✅ Result: Instead of {len(unique_pages)} individual page visits,")
    print(f"   you now have {len(final_unique_pages)} category-based pages representing")
    print(f"   the core functionality of your app!")


def demonstrate_usage_in_existing_code():
    """Show how to integrate into existing workflow"""
    print(f"\n💡 Integration Example:")
    print("=" * 40)
    
    code_example = '''
# In your existing chain processing code:

from chain_understand import apply_category_based_page_merging

async def process_app_navigation(chain_data, task_description):
    """Process app navigation with category-based merging"""
    
    # Apply category-based merging instead of standard merging
    updated_chain, merge_report = await apply_category_based_page_merging(
        chain=chain_data,
        task_info=task_description
    )
    
    # Check the results
    pages_deleted = merge_report['summary']['pages_physically_deleted']
    categories = len(merge_report['summary']['categories_identified'])
    
    print(f"✅ Merged into {categories} page categories")
    print(f"🗑️  Deleted {pages_deleted} duplicate page nodes")
    
    return updated_chain, merge_report

# Usage:
chain, report = await process_app_navigation(
    chain_data=my_triplet_chain,
    task_description="Shopping for electronics"
)
'''
    
    print(code_example)


if __name__ == "__main__":
    print("🚀 Category-Based Page Merging Demo")
    print("This demo shows how to merge pages by functionality categories")
    print("instead of keeping every individual page visit.\n")
    
    # Run the demonstration
    asyncio.run(demonstrate_category_merging())
    
    # Show integration example
    demonstrate_usage_in_existing_code()
    
    print(f"\n🎯 Perfect for:")
    print("   • E-commerce apps (Home, Listing, Product Detail)")
    print("   • Social media apps (Feed, Profile, Settings)")
    print("   • News apps (Headlines, Article, Categories)")
    print("   • Any app with repeated page types!")
