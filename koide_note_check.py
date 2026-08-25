# -*- coding: utf-8 -*-
"""End-to-end consistency check on the preprint note."""
import re, subprocess, sys, os
P='/mnt/user-data/outputs/preprint/tauon_conventions_note.tex'
raw=re.sub(r'(?m)^%.*$','',open(P,encoding='utf-8').read())
body,bib=raw.split(r'\begin{thebibliography}')
flat=re.sub(r'\s+',' ',body)
fails=[]
def chk(c,m):
    print(('  PASS  ' if c else '  FAIL  ')+m)
    if not c: fails.append(m)

print('='*74); print('1. STRUCTURE AND REFERENCES'); print('='*74)
lab={m.group(1):m.start() for m in re.finditer(r'\\label\{([^}]+)\}',body)}
refs=[m.group(1) for m in re.finditer(r'\\(?:ref|eqref)\{([^}]+)\}',body)]
chk(all(r in lab for r in refs), 'every reference resolves  %s'%[r for r in refs if r not in lab])
fwd=[r for m,r in ((m,m.group(1)) for m in re.finditer(r'\\(?:ref|eqref)\{([^}]+)\}',body))
     if r in lab and lab[r]>m.start()]
print('  INFO  %d of %d references are forward (roadmap pointers)'%(len(fwd),len(refs)))
secs=[m.group(2) for m in re.finditer(r'\\(section|subsection)\*?\{([^}]*)\}',body)]
print('  INFO  sections: %s'%' | '.join(secs))

print('\n'+'='*74); print('2. NUMBERS AGREE THROUGHOUT'); print('='*74)
NUM={'3447':'published tauon','3483':'proposed tauon','97':'proposed E_s',
     '96':'shell edges','120':'centred D4 edges','36':'muon E_s / F_square',
     '216':'muon static','3492':'tauon static','0.663084':'static K',
     '0.666724':'shed K','129':'pion supports','3495':'static row'}
for k,v in NUM.items():
    chk(k in flat.replace(',',''), '%-9s %s'%(k,v))
chk('3481' in flat, 'Koide demand 3481 stated')
import math
chk(97*36-9==3483 and 36*6-9==207 and 96*36-9==3447, 'arithmetic internally consistent')

print('\n'+'='*74); print('3. THE REFRAMING IS CONSISTENT'); print('='*74)
chk('is not a convention at all' in flat, 'abstract says Convention A dissolves')
chk('apparent rather than real' in flat, 'body says the departure is apparent')
chk('A defect occupies the object it occupies' in flat, 'the support argument is stated')
chk('96 or 120' not in flat, 'the old framing ("96 or 120") is gone')
chk('Nothing in the framework says why one' not in flat, 'the old objection is gone')

print('\n'+'='*74); print('4. CLAIMS ARE SCOPED'); print('='*74)
abs_=flat[flat.index('Koide')-200:flat.index('What is claimed')]
chk('is unchanged at 3447' in abs_, 'the ABSTRACT states the published prediction stands')
chk(flat.count('stands at $C_\\tau = 3447$')==0, 'the closing disclaimer is not repeated')
chk('candidate resolutions of an open problem, not corrections' in flat, 'abstract disclaims correction')
chk('one non-trivial application' in flat, 'weakness stated')
chk('Within the description set out here' in flat, 'kinematic claim is scoped')
chk(flat.count('this description')<=1, 'the scope is stated once, not repeated')
chk('disjoint' in flat, 'the H3 tension is stated')

print('\n'+'='*74); print('5. ATTRIBUTION'); print('='*74)
for t in ['Axiom~4','Axiom~3','Landauer','active sector','H3','H4']:
    ms=list(re.finditer(re.escape(t),flat))
    if not ms: continue
    w=flat[max(0,ms[0].start()-230):ms[0].start()+230]
    chk('\\cite' in w, 'first use of %-14s carries a citation'%t)
keys=set(re.findall(r'\\bibitem\{([^}]+)\}',bib))
cited=set()
for g in re.findall(r'\\cite\{([^}]+)\}',body): cited|={c.strip() for c in g.split(',')}
chk(keys==cited, 'bibliography matches citations  %s'%(keys^cited))

print('\n'+'='*74); print('6. BUILD AND SCRIPTS'); print('='*74)
d=os.path.dirname(P)
for _ in range(2):
    subprocess.run(['pdflatex','-interaction=nonstopmode','tauon_conventions_note.tex'],
                   cwd=d,capture_output=True)
log=open(os.path.join(d,'tauon_conventions_note.log'),errors='ignore').read()
chk('undefined' not in log.lower(),'no undefined references or citations')
m=re.search(r'Output written on \S+ \((\d+) pages',log)
print('  INFO  %s pages'%(m.group(1) if m else '?'))
for sc in ('koide_17_enclosed_voids.py','koide_18_a2_angles.py'):
    r=subprocess.run([sys.executable,sc],cwd=d,capture_output=True,timeout=900)
    chk(r.returncode==0,'%s exits 0'%sc)
for f in ('koide_fig5_enclosed_void.pdf','koide_fig6_faces_circuits.pdf'):
    chk(os.path.exists(os.path.join(d,f)),'figure present: %s'%f)

print('\n'+'='*74); print('RESULT: %d failures'%len(fails))
for f in fails: print('   -',f)
