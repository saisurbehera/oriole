#!/usr/bin/env python3
"""Debug XML parsing"""

import xml.etree.ElementTree as ET

xml_path = "data/nethackwiki_current.xml"

# Parse first few elements
context = ET.iterparse(xml_path, events=('start', 'end'))
context = iter(context)

count = 0
for event, elem in context:
    if count > 100:
        break
        
    if event == 'end':
        print(f"Event: {event}, Tag: {elem.tag}, Text: {elem.text[:50] if elem.text else 'None'}")
        
        if 'page' in elem.tag:
            print("  Found page element!")
            # Look for title and text
            for child in elem:
                print(f"    Child: {child.tag}")
                if 'title' in child.tag:
                    print(f"      Title: {child.text}")
                elif 'revision' in child.tag:
                    for subchild in child:
                        if 'text' in subchild.tag:
                            print(f"      Text length: {len(subchild.text) if subchild.text else 0}")
                            
    count += 1