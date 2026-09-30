"""Export individual manuscript and supplementary panels from computed results."""
from pathlib import Path
import argparse,hashlib,json,shutil
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import patheffects
from matplotlib.font_manager import FontProperties
from matplotlib.path import Path as MplPath
from matplotlib.patches import FancyArrowPatch,Rectangle,Polygon,Ellipse
from scipy.ndimage import gaussian_filter
from scipy.spatial import ConvexHull
from PIL import Image
import f2,f3,f4,f5,correlation,supplementary
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'outputs/figures'
MAIN='D4_K10_L9_U47'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def prepare():
    specs=json.loads(Path(__file__).with_name('source_paths.json').read_text())
    for name,spec in specs.items():
        folder=ROOT/'outputs/figure_sources'/name;folder.mkdir(parents=True,exist_ok=True);manifest={'files':{}}
        for role,relative in spec['files'].items():
            source=ROOT/relative
            if not source.is_file():raise FileNotFoundError(f'Missing computed figure input: {relative}')
            if spec['kind']=='canonical':manifest['files'][role]={'path_from_0_REFIT':relative,'sha256':sha(source)}
            else:
                target=folder/'inputs'/role;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(source,target)
                manifest['files'][role]={'sha256':sha(source),'source':{'source_path':relative},'canonical_path_from_0_REFIT':relative}
        where=folder/'sources.json' if spec['kind']=='canonical' else folder/'inputs'/('provenance.json' if name=='f4' else 'sources.json')
        where.write_text(json.dumps(manifest,indent=2)+'\n')
    for m in [f2,f3,f4,f5]:
        for k,v in dict(np=np,pd=pd,plt=plt,matplotlib=matplotlib,gaussian_filter=gaussian_filter,ConvexHull=ConvexHull,Polygon=Polygon,Ellipse=Ellipse,Image=Image,FontProperties=FontProperties,MplPath=MplPath,FancyArrowPatch=FancyArrowPatch,Rectangle=Rectangle,patheffects=patheffects).items():setattr(m,k,v)
def single(path,draw,dpi,size=(5.8,4.7)):
    path.parent.mkdir(parents=True,exist_ok=True);fig,ax=plt.subplots(figsize=size,layout='constrained');draw(ax)
    fig.savefig(path,dpi=dpi,facecolor='white');plt.close(fig)
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--dpi',type=int,default=300);a=ap.parse_args()
    if a.dpi<100:raise ValueError('Use at least 100 dpi.')
    prepare();OUT.mkdir(parents=True,exist_ok=True);(OUT/'plot_data').mkdir(exist_ok=True)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.linewidth':.65})
    events,links,pairs,summary,projections,manifest=f2.load_and_validate();field=f2.grid_and_flow(events,links)
    field[-1].to_csv(OUT/'plot_data/flow_grid_Dim0_Dim3.csv',index=False)
    single(OUT/'main/Figure2_macro_projection.png',lambda ax:f2.draw_geometry(ax,events,letter=None),a.dpi)
    single(OUT/'main/Figure2_macro_distance_probability.png',lambda ax:f2.draw_distance(ax,pairs,letter=None),a.dpi)
    single(OUT/'main/Figure2_observed_displacement.png',lambda ax:f2.draw_flow(ax,field,letter=None),a.dpi)
    expected=pd.read_csv(ROOT/'outputs/field_tests/flow_grid_Dim0_Dim3.csv')
    velocities=[expected[f'null_expected_mean_delta_Dim{d}'].to_numpy().reshape(25,25) for d in [0,3]]
    expected_field=(field[0],field[1],field[2],velocities,field[4],field[5])
    def draw_expected(ax):
        f2.draw_flow(ax,expected_field,letter=None);ax.set_title('Expected displacement under randomized order',loc='left',fontsize=10.5)
    single(OUT/'main/Figure2_expected_displacement.png',draw_expected,a.dpi)
    effects=pd.read_csv(ROOT/'outputs/event_distance/participant_and_pooled_effects.csv')
    labels=[f'{p}  ({n} links)' for p,n in zip(effects.participant,effects.links)]
    positions=np.r_[np.arange(5),6.1]
    with plt.rc_context({'font.size':10.5,'axes.spines.top':False,'axes.spines.right':False}):
        fig,ax=plt.subplots(figsize=(8.0,4.8));fig.subplots_adjust(left=.34,right=.96,bottom=.18,top=.84)
        for y,r in zip(positions,effects.itertuples(index=False)):
            ax.plot([r.observed_mean_distance,r.null_expected_mean_exact],[y,y],color='#CBD2D5',linewidth=2.2,zorder=1)
        ax.scatter(effects.observed_mean_distance,positions,color='#246579',s=48,zorder=3,label='Observed')
        ax.scatter(effects.null_expected_mean_exact,positions,color='white',edgecolors='#59676E',s=48,linewidths=1.3,zorder=3,label='Expected under shuffle')
        ax.axhline(5.2,color='#D0D5D8',linewidth=.8);ax.set_yticks(positions,labels);ax.set_ylim(6.8,-.8)
        ax.set_xlabel('Mean consecutive-event distance in 4-D space');ax.set_title('Observed and expected distances',loc='left',fontsize=13,pad=29)
        ax.legend(loc='lower left',bbox_to_anchor=(0,1),ncol=2,frameon=False,fontsize=9,borderaxespad=0)
        ax.tick_params(axis='y',length=0);ax.spines['left'].set_visible(False);ax.margins(x=.1)
        fig.savefig(OUT/'main/Figure2_event_distance.png',dpi=a.dpi,facecolor='white');plt.close(fig)
    # Basin map and support retain the reviewed rendering functions.
    ev,summary3,support,manifest3=f3.load_inputs();limits=f3.projection_limits(ev,(0,3))
    def basin_map(ax):
        f3.draw_map(ax,ev,MAIN,(0,3),limits,'4-D 10-Basin')
        f3.add_basin_labels(ax.figure,ax,ev,MAIN,(0,3),'DejaVu Sans')
    single(OUT/'main/Figure3_basin_projection.png',basin_map,a.dpi)
    single(OUT/'main/Figure3_basin_occupancy.png',lambda ax:f3.draw_counts(ax,support,MAIN,'Basin-state occupancy'),a.dpi,size=(5.8,3.2))
    primary=pd.read_csv(ROOT/f'outputs/downstream/local/{MAIN}/feature_primary3/pairwise_basin_similarity.csv')
    def feature_routing(ax):
        x=primary.feature_profile_similarity_cosine;y=primary.routing_profile_similarity_cosine;ax.scatter(x,y,s=29,c='#22689E',edgecolors='white',linewidths=.4);m,b=np.polyfit(x,y,1);xx=np.array([x.min(),x.max()]);ax.plot(xx,m*xx+b,c='#50585E',lw=1)
        ax.set(xlabel='Feature-profile cosine similarity',ylabel='One-step routing-profile cosine similarity');ax.spines[['top','right']].set_visible(False)
    single(OUT/'main/Figure3_feature_routing.png',feature_routing,a.dpi)
    data4,manifest4,checked=f4.validate_and_load();d=data4[MAIN]
    f4.draw_empirical_3d(d,'Dim2',OUT/'main/Figure4_empirical_network.png',a.dpi)
    f4.draw_context_similarity(d,OUT/'main/Figure4_context_similarity.png',a.dpi)
    for context in ['social','physical']:f4.draw_network(d,context,OUT/f'main/Figure4_{context}_routing.png',a.dpi,None,'Context-conditioned routing: '+context.title())
    f4.draw_contrast(d,OUT/'main/Figure4_pathway_engagement.png',a.dpi)
    data5,profiles,orders,manifest5,audit=f5.load_inputs();d=data5[MAIN]
    for kind,name,heading in [('mca','representational_projection','Representational projection'),('pca','successor_projection','Successor projection')]:
        xy=d['centroids'][['Dim0','Dim3']].to_numpy() if kind=='mca' else d['pca'].iloc[:,:2].to_numpy();bounds=f5.common_bounds([xy],.18)
        def draw(ax,kind=kind,bounds=bounds,heading=heading):
            ax,points,labels=f5.draw_map(ax,d,kind,bounds,heading);supplementary.basin_labels(ax,points,labels)
        single(OUT/f'main/Figure5_{name}.png',draw,a.dpi)
    ab=f5.common_bounds([d['aligned'][['terrain_x','terrain_y']].to_numpy(),d['aligned'][['predictive_x','predictive_y']].to_numpy()],.22)
    def alignment(ax):
        ax,points,labels=f5.draw_alignment(ax,d,ab);supplementary.basin_labels(ax,points,labels)
    single(OUT/'main/Figure5_procrustes_alignment.png',alignment,a.dpi)
    pairs5=d['pairs'];lim=((0,pairs5.geometry_distance.max()*1.08),(0,pairs5.predictive_distance.max()*1.08))
    single(OUT/'main/Figure5_complete_profile_distance.png',lambda ax:f5.draw_distance(ax,d,lim),a.dpi)
    paths={'successor_correlations':ROOT/'outputs/downstream/local/comparisons/successor_geometry_all_assignments.csv','assignment_overview':ROOT/'outputs/downstream/local/comparisons/assignment_overview.csv','cross_solution_components':ROOT/'outputs/cross_solution/partial_conjunction/component_results.csv','across_geometry_holm':ROOT/'outputs/cross_solution/partial_conjunction/assignment_holm_results.csv'}
    panel=correlation.prepare_data(paths,np,pd);image,font,supported=correlation.make_figure(panel,np,matplotlib,plt,Image)
    (OUT/'main/Figure5_all_solutions.png').write_bytes(image);(OUT/'Figure5_correlation_legend.txt').write_text(correlation.legend_text())
    paths={k:ROOT/v for k,v in json.loads(Path(__file__).with_name('supplementary_paths.json').read_text()).items()}
    r=supplementary.Renderer(paths,OUT/'supplementary',a.dpi,'DejaVu Sans');r.run()
    (OUT/'supplementary/INTERNAL_CAPTIONS.txt').unlink(missing_ok=True)
    shutil.copyfile(ROOT/'documentation/Supplementary_Figure_Legends.md',OUT/'supplementary/LEGENDS.md')
    panel_manifest={'main':[str(p.relative_to(OUT)) for p in sorted((OUT/'main').glob('*.png'))],'supplementary':[str(p.relative_to(OUT)) for p in sorted((OUT/'supplementary/panels').glob('*.png'))],'panel_letters_embedded':False,'manual_assembly':'Figure 1 is conceptual. Combine Figure 5 projection/alignment exports manually.'}
    if len(panel_manifest['main'])!=18 or len(panel_manifest['supplementary'])!=25:raise ValueError('Unexpected figure-panel count.')
    (OUT/'PANEL_MANIFEST.json').write_text(json.dumps(panel_manifest,indent=2)+'\n')
if __name__=='__main__':main()
