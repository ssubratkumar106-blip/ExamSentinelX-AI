import os
import re

TEMPLATE_DIR = r"c:\Users\DELL\OneDrive\Desktop\XtragradIntern\ExamSentinelX AI\frontend\templates"

# Map of colors to replace to remove purple
color_replacements = {
    # Indigo/Purple hexes
    r'#6366f1': '#3B82F6', # accent -> blue
    r'#8b5cf6': '#10B981', # purple -> green
    r'#818cf8': '#60A5FA', 
    r'#c084fc': '#F59E0B', 
    # Indigo/Purple rgba
    r'99,\s*102,\s*241': '59, 130, 246',
    r'139,\s*92,\s*246': '16, 185, 129',
}

# Sidebar modifications for student wrap
sidebar_replacement_regex = r'<aside class="student-sidebar">(.*?)</aside>'
def process_file(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()

    original_content = content
    for old, new in color_replacements.items():
        content = re.sub(old, new, content, flags=re.IGNORECASE)

    if content != original_content:
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"Updated {filepath}")

for root, dirs, files in os.walk(TEMPLATE_DIR):
    for file in files:
        if file.endswith('.html'):
            filepath = os.path.join(root, file)
            # Skip base.html and exam.html because we already manually perfected them
            if file in ['base.html', 'exam.html']:
                continue
            process_file(filepath)
