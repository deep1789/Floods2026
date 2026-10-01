import re
txt='\n'.join(open(f'parts/p{i}.md').read() for i in (1,2,3,4))
txt=re.sub(r'\{\{T:(\w+)\}\}',lambda m:open(f'tables/{m.group(1)}.md').read(),txt)
open('PAPER_PLAN.md','w').write(txt)
body=re.sub(r'```.*?```','',txt,flags=re.S); body=re.sub(r'\$\$.*?\$\$','',body,flags=re.S); body=re.sub(r'\|.*\|','',body); body=re.sub(r'!\[.*?\]\(.*?\)','',body)
print('words (prose only, no tables/code/display math):',len(body.split()),' total tokens incl everything:',len(txt.split()))
print('unfilled placeholders:',re.findall(r'\{\{.*?\}\}',txt))
