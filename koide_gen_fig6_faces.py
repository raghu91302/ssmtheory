#!/usr/bin/env python3
"""Figure 2 of the note: the cuboctahedron's squares are 2-faces; the
24-cell's are only circuits in the edge graph, spanned by four triangles."""
import numpy as np, matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
fig,ax=plt.subplots(1,2,figsize=(9.2,4.2))
for a in ax: a.set_aspect('equal'); a.axis('off'); a.set_xlim(-1.5,1.5); a.set_ylim(-1.7,1.5)

s=0.85
sq=np.array([[-s,-s],[s,-s],[s,s],[-s,s]])
ax[0].add_patch(plt.Polygon(sq,facecolor='#AFA9EC',alpha=0.55,edgecolor='#3C3489',lw=1.6))
ax[0].scatter(sq[:,0],sq[:,1],s=70,c='#7F77DD',edgecolors='#26215C',zorder=3,linewidths=0.7)
ax[0].set_title("cuboctahedron\nthe square is a 2-face",fontsize=11,pad=10)
ax[0].text(0,0,"filled",ha='center',va='center',fontsize=10,color='#26215C')
ax[0].text(0,-1.35,"6 squares, all of them faces\nidentifying them would quotient the complex",
           ha='center',fontsize=9,color='#444441')

ax[1].add_patch(plt.Polygon(sq,facecolor='none',edgecolor='#D85A30',lw=1.6,ls=(0,(4,2))))
ax[1].scatter(sq[:,0],sq[:,1],s=70,c='#D85A30',edgecolors='#712B13',zorder=3,linewidths=0.7)
mid=np.array([[0,-s],[s,0],[0,s],[-s,0]])
for i in range(4):
    ax[1].plot([sq[i,0],mid[i,0]],[sq[i,1],mid[i,1]],color='#E8A98F',lw=0.9,zorder=1)
    ax[1].plot([mid[i,0],sq[(i+1)%4,0]],[mid[i,1],sq[(i+1)%4,1]],color='#E8A98F',lw=0.9,zorder=1)
ax[1].scatter(mid[:,0],mid[:,1],s=26,c='#E8A98F',edgecolors='#993C1D',zorder=2,linewidths=0.5)
ax[1].set_title("24-cell\nthe square is only a circuit",fontsize=11,pad=10)
ax[1].text(0,0,"not a cell",ha='center',va='center',fontsize=10,color='#712B13')
ax[1].text(0,-1.35,"72 circuits, none a face; each spanned by 4 triangles\nidentifying them leaves the complex unchanged",
           ha='center',fontsize=9,color='#444441')
plt.tight_layout(); plt.savefig('koide_fig6_faces_circuits.pdf',bbox_inches='tight')
plt.savefig('koide_fig6_faces_circuits.png',dpi=150,bbox_inches='tight')
print("wrote koide_fig6_faces_circuits.pdf")
