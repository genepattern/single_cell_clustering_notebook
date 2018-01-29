from copy import deepcopy
from decimal import Decimal

import ipywidgets as widgets
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as st
import seaborn as sns
import traitlets
from ipywidgets import HTML, Layout
from matplotlib.colorbar import ColorbarBase
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import MaxNLocator
from statsmodels.sandbox.stats.multicomp import multipletests

import plotly.offline as py
import plotly.tools as tls
import scanpy.api as sc

from beakerx import *

py.init_notebook_mode()

_MARKER_GENE = ''
_FLEX_CENTER_LAYOUT = Layout(display='flex', justify_content='center', align_items='center')
# -------------------- HELPERS --------------------


def _create_progress_bar():
    return HTML('<progress></progress>', layout=Layout(width='200px', height='20px'))


def _plot_placeholder():
    return HTML('<p>Plots will display here.</p>')


# -------------------- LOAD DATASET --------------------


def load_dataset(input_type, path, min_cells=3, min_genes=200):
    data = _load_dataset(input_type, path, min_cells, min_genes)
    _load_dataset_ui(data)
    return data


def _load_dataset(input_type, path, min_cells, min_genes):

    if input_type == 'tabbed':
        data = sc.read_text(path).transpose()
    elif input_type == '_10x':
        data = sc.read(path + 'matrix.mtx', cache=True).transpose()
        data.var_names = np.genfromtxt(path + 'genes.tsv', dtype=str)[:, 1]
        data.obs_names = np.genfromtxt(path + 'barcodes.tsv', dtype=str)
    else:
        display(HTML('Unknown file type <code>{}</code>.'.format(input_type)))
        return
    # Load data

    # Basic filtering.
    sc.pp.filter_cells(data, min_genes=min_genes)
    sc.pp.filter_genes(data, min_cells=min_cells)

    # Plot some information about mitochondrial genes, important for quality control
    mito_genes = [name for name in data.var_names if name.startswith('MT-')]
    data.obs['percent_mito'] = np.sum(
        data[:, mito_genes].X, axis=1) / np.sum(
            data.X, axis=1)
    # add the total counts per cell as observations-annotation to data
    data.obs['n_counts'] = np.sum(data.X, axis=1)
    data.is_log = False
    return data


def _load_dataset_ui(data):
    measures = [data.obs['n_genes'], data.obs['n_counts'], data.obs['percent_mito']]
    measure_names = ['# of Genes', 'Total Counts', '% Mitochondrial Genes']

    # Violin plots of single variables
    sns.set_style("whitegrid")
    fig1 = plt.figure(1, figsize=(12, 7))
    subplot_num = 131
    for m, m_name in zip(measures, measure_names):
        ax = plt.subplot(subplot_num)
        ax = sns.violinplot(data=m, inner=None, color="0.8", ax=ax)
        ax = sns.stripplot(data=m, jitter=True, ax=ax)
        ax.set_xlabel(m_name, size=16)
        ax.set_xticklabels([''])
        subplot_num += 1
    plt.subplots_adjust(wspace=0.5)
    plt.close()

    fig1_out = widgets.Output()
    with fig1_out:
        display(fig1)

    # Scatter plots of paired variables
    fig2 = plt.figure(2, figsize=(12, 5), dpi=100)
    ax1 = plt.subplot(121)
    ax1 = sns.regplot(x='Total Counts', y='# of Genes', data=pd.DataFrame(
        [measures[1], measures[0]], index=['Total Counts', '# of Genes']).T, fit_reg=False, ax=ax1)
    ax1.set_xlabel(ax1.get_xlabel(), size=14)
    ax1.set_ylabel(ax1.get_ylabel(), size=14)
    ax2 = plt.subplot(122)
    ax2 = sns.regplot(x='Total Counts', y='% Mitochondrial Genes', data=pd.DataFrame(
        [measures[1], measures[2]], index=['Total Counts', '% Mitochondrial Genes']).T, fit_reg=False, ax=ax2)
    ax2.set_xlabel(ax2.get_xlabel(), size=14)
    ax2.set_ylabel(ax2.get_ylabel(), size=14)
    plt.subplots_adjust(wspace=0.3)
    plt.close()

    fig2_out = widgets.Output()

    def fig2_out_callback(e):
        fig2_out.clear_output()
        with fig2_out:
            py.iplot_mpl(fig2, show_link=False)
    fig2_out.on_displayed(fig2_out_callback)

    # Descriptive text
    header = HTML(
        '''<p>Loaded <code>{}</code> cells and <code>{}</code> total genes.</p>
           <p>We can identify and exclude outlier cells based on some quality control metrics.</p>'''.format(len(measures[0]), len(data.var_names)))
    # Parent container
    tabs = widgets.Tab(children=[fig1_out, fig2_out])
    tabs.set_title(0, 'Single Variables')
    tabs.set_title(1, 'Pairwise Variables')
    display(header, tabs)

# -------------------- PREPROCESS COUNTS --------------------


def preprocess_counts(data, min_n_genes='-inf', max_n_genes='inf', min_percent_mito='-inf', max_percent_mito='inf', normalization_method='LogNormalize'):
    n_genes_range = [float(min_n_genes), float(max_n_genes)]
    percent_mito_range = [float(min_percent_mito), float(max_percent_mito)]

    data, data_raw, pc_sdev = _preprocess_counts(data, n_genes_range, percent_mito_range, normalization_method)
    _preprocess_counts_ui(data, data_raw, pc_sdev)
    return data


def _preprocess_counts(data, n_genes_range, percent_mito_range, normalization_method):
    if data.raw:
        display(HTML('This data has already been preprocessed. Please run <code>load_dataset()</code> again if you would like to perform preprocessing again.'))
    else:
        # Remove cells that have too many mitochondrial genes expressed or too many total counts.
        n_genes_filter = (data.obs['n_genes'] > n_genes_range[0]) & (data.obs['n_genes'] < n_genes_range[1])
        if not n_genes_filter.any():
            data = data[n_genes_filter, :]

        percent_mito_filter = (data.obs['percent_mito'] > percent_mito_range[0]) & (
            data.obs['percent_mito'] < percent_mito_range[1])
        if not percent_mito_filter.any():
            data = data[percent_mito_filter, :]

        # Set the `.raw` attribute of AnnData object to the logarithmized raw gene
        # expression for later use in differential testing and visualizations of
        # gene expression. This simply freezes the state of the data stored in
        # `data_raw`.
        if normalization_method == 'LogNormalize' and data.is_log is False:
            data_raw = sc.pp.log1p(data, copy=True)
            data.raw = data_raw
        else:
            data.raw = deepcopy(data.X)

        # Per-cell normalize the data matrix $\mathbf{X}$, identify highly-variable genes and compute logarithm.
        sc.pp.normalize_per_cell(data, counts_per_cell_after=1e4)
        filter_result = sc.pp.filter_genes_dispersion(
            data, min_mean=0.1, max_mean=8, min_disp=1)

        # Logarithmize the data.
        if normalization_method == 'LogNormalize' and data.is_log is False:
            sc.pp.log1p(data)
            data.is_log = True

        # Regress out effects of total counts per cell and the percentage of
        # mitochondrial genes expressed. Scale the data to unit variance.
        sc.pp.regress_out(data, ['n_counts', 'percent_mito'])
        sc.pp.scale(data, max_value=10)

    # Calculate PCA regardless
    data, pc_sdev = _run_pca(data)
    # Save the result.
    return data, data.raw.X, pc_sdev


def _preprocess_counts_ui(data, data_raw, pc_sdev):
    cell_text = HTML(
        '<p><code>{}/{}</code> cells passed filtering.</p>'.format(len(data.obs_names), data_raw.shape[0]))
    v_genes_text = HTML(
        '<p><code>{}/{}</code> genes detected as variable genes.</p>'.format(len(data.var_names), data_raw.shape[1]))
    if data.is_log:
        log_text = HTML('<p>Data is log normalized.</p>')
    else:
        log_text = HTML('<p>Data is not normalized.</p>')
    regress_text = HTML(
        '''<p>Performed batch effect removal based on:</p><ol><li># of detected molecules per cell</li>
        <li>% mitochondrial gene content</li></ol>''')

    display(cell_text, v_genes_text, log_text, regress_text)
    pca_fig = _plot_pca(pc_sdev)
    py.iplot(pca_fig, show_link=False)

# -------------------- PERFORM CLUSTERING & MARKER ANALYSIS --------------------


def perform_clustering_analysis(data):

    # -------------------- tSNE PLOT --------------------
    pc_sdev = pd.Series(np.std(data.obsm['X_pca'], axis=0))
    pc_sdev.index = pc_sdev.index + 1

    pcs = 10
    resolution = 1.3
    perplexity = 13

    # Parameter values
    pc_range = range(2, len(pc_sdev) + 1)
    res_range = [float('{:0.1f}'.format(x))
                 for x in list(np.arange(.1, 3.1, 0.1))]
    perp_range = range(5, min(100, len(data.obs_names)))

    # Parameter slider widgets
    pc_slider = widgets.SelectionSlider(
        options=pc_range, value=pcs, description="# of PCs", continuous_update=False)
    res_slider = widgets.SelectionSlider(
        options=res_range, value=resolution, description="Resolution", continuous_update=False)
    perp_slider = widgets.SelectionSlider(
        options=perp_range, value=perplexity, description="Perplexity", continuous_update=False)

    # Output widget
    plot_output = widgets.Output(layout=Layout(height='700px', display='flex',
                                               align_items='center', justify_content='center'))
    with plot_output:
        display(HTML('<p>Plot will display here.</p>'))
    markers_ui_box = widgets.Box([_create_progress_bar()])

    # "Go" button to plot on click
    def plot_tsne_callback(button):
        plot_output.clear_output()
        progress_bar = _create_progress_bar()

        with plot_output:
            # show progress bar
            display(progress_bar)

            # perform tSNE calculation and plot
            _run_tsne(data, pc_slider.value, res_slider.value, perp_slider.value)
            py_fig = _plot_tsne(data)
            py.iplot(py_fig, show_link=False)

            # close progress bar
            progress_bar.close()

        # tSNE and violin plots
        # tsne_marker_plots_tab = widgets.Tab()
        # violin_marker_plots_tab = widgets.Tab()
        # markers_ui_box.children = _markers_ui(data, tsne_marker_plots_tab, violin_marker_plots_tab)
        markers_ui_box.children = [_markers_ui_new(data)]

    # Button widget
    go_button = widgets.Button(description='Plot')
    go_button.on_click(plot_tsne_callback)

    # Parameter descriptions
    pc_info = HTML('''<h4>Number of PCs (Principal Components)</h4><p>The number of selected principal components to use in clustering. Determine the number of principal components (PCs) to use by drawing a cutoff where there is a clear elbow in the graph.</p>
                      <h4>Resolution</h4><p>The resolution parameter sets the ‘granularity’ of the downstream clustering, with increased values leading to a greater number of clusters. We find that setting this parameter between 0.6-1.2 typically returns good results for single cell datasets of around 3K cells. Optimal resolution often increases for larger datasets.</p>
                      <h4>Perplexity</h4><p>The perpelexity parameter loosely models the number of close neighbors each point has. More info on how perplexity matters <a href="https://distill.pub/2016/misread-tsne/">here</a>.</p>''')

    param_header = HTML('<h4>Parameters</h4>')

    param_info = widgets.Accordion([pc_info])
    param_info.set_title(0, 'Parameter Info')
    param_info.selected_index = None
    sliders = widgets.HBox([pc_slider, res_slider, perp_slider])

    ui = widgets.VBox([param_header, param_info, sliders, go_button])

    plot_box = widgets.VBox([ui, plot_output])
    tabs = widgets.Tab([plot_box, markers_ui_box])

    tabs.set_title(0, '1. Visualize Clusters')
    tabs.set_title(1, '2. Explore Markers')
    display(tabs)

    return data


def _run_pca(data):
    sc.tl.pca(data)
    data.obsm['X_pca'] *= -1  # multiply by -1 to match Seurat R
    pc_sdev = pd.Series(np.std(data.obsm['X_pca'], axis=0))
    pc_sdev.index = pc_sdev.index + 1
    return data, pc_sdev


def _plot_pca(pc_sdev):
    elbow_output = widgets.Output()
    pcs = 10
    resolution = 1.3
    perplexity = 13

    # mpl figure
    fig_elbow_plot = plt.figure(figsize=(4.5, 5), dpi=100)
    plt.plot(pc_sdev, 'o')
    ax = fig_elbow_plot.gca()
    x_axis = ax.axes.get_xaxis()
    y_axis = ax.axes.get_yaxis()
    x_axis.set_major_locator(MaxNLocator(integer=True))
    y_axis.set_major_locator(MaxNLocator(integer=True))
    ax.set_xlabel('PC', size=16)
    ax.set_ylabel('Standard Deviation of PC', size=14)
    plt.close()

    # plot interactive
    py_fig = tls.mpl_to_plotly(fig_elbow_plot)
    py_fig['layout']['margin'] = {'l': 35, 'r': 14, 't': 10, 'b': 45}

    return py_fig


def _run_tsne(data, pcs, resolution, perplexity):
    sc.tl.tsne(data, n_pcs=pcs, perplexity=perplexity, learning_rate=1000, n_jobs=8)
    sc.tl.louvain(data, n_neighbors=10, resolution=resolution, recompute_graph=True)

    return data


def _plot_tsne(data):
    cell_clusters = data.obs['louvain_groups'].astype(int)
    cluster_names = np.unique(cell_clusters).tolist()
    num_clusters = len(cluster_names)
    tsne_coordinates = pd.DataFrame(data.obsm['X_tsne'], columns=['tSNE_1', 'tSNE_2'])

    palette = sns.color_palette("Set2", num_clusters)
    colors = dict(zip(cluster_names, palette))
    cluster_colors = cell_clusters.map(dict(zip(cluster_names, palette))).tolist()

    df = tsne_coordinates
    df['colors'] = cell_clusters.tolist()

    fig, ax = plt.subplots(1, 1, figsize=(8, 6.5), dpi=100)
    for c, group in df.groupby(by='colors'):
        group.plot(x='tSNE_1', y='tSNE_2', kind='scatter', c=colors[c], label=c, alpha=0.7, ax=ax, legend=False)
        # ax = sns.regplot(x='tSNE_1', y='tSNE_2', data=group, color=colors[c], fit_reg=False, label=c)
    # ax.legend(loc='best')
    # ax = sns.regplot(x='tSNE_1', y='tSNE_2', data=tsne_coordinates,
    #                  scatter_kws={'color': cluster_colors}, fit_reg=False)
    plt.title('tSNE Visualization', size=16)
    ax.set_xlabel(ax.get_xlabel(), size=14)
    ax.set_ylabel(ax.get_ylabel(), size=14)
    plt.tight_layout()
    plt.close()

    py_fig = tls.mpl_to_plotly(fig)
    py_fig['layout']['showlegend'] = True
    py_fig.update(data=[dict(name=c) for c in cluster_names])

    return py_fig


def _markers_ui_new(data):

    # Commonly used data
    cell_clusters = data.obs['louvain_groups'].astype(int)
    cluster_names = np.unique(cell_clusters).tolist()
    cluster_names.sort(key=int)

    # Initialize output widgets for scope
    marker_plot_output = widgets.Output(layout=Layout(height='500px'))
    violin_plot_output = widgets.Output(layout=Layout(height='200px'))
    cluster_table_output = widgets.Output(layout=Layout(height='300px', overflow_y='auto'))
    main_table_output = widgets.Output()

    # Create UI container elements
    main_box = widgets.VBox(layout=Layout(width='57%'))
    main_header_box = widgets.HBox()
    main_box.children = [main_header_box, marker_plot_output]

    sidebar_box = widgets.VBox(layout=Layout(width='43%'))
    cluster_table_header_box = widgets.VBox()
    cluster_table_box = widgets.VBox([cluster_table_header_box, cluster_table_output])
    sidebar_box.children = [violin_plot_output, cluster_table_box]

    # TODO make this table output
    main_table_box = widgets.Box([main_table_output])

    # ------------------------- Output Placeholders -------------------------



    # Fill container elements
    # ------------------------- Main header -------------------------
    gene_input = widgets.Text()
    update_button = widgets.Button(description='Plot Expression')
    main_header_box.children = [gene_input, update_button]

    def check_gene_input(t):
        '''Don't allow submission of empty input.'''
        if gene_input.value == '':
            update_button.disabled = True
        else:
            update_button.disabled = False

    def update_plots(b):
        # Get gene expression
        gene = gene_input.value
        gene = str.strip(gene).capitalize()
        gene_loc = data.raw.var_names.get_loc(gene)

        if type(data.raw.X) in [np.array, np.ndarray]:
            gene_values = pd.Series(data.raw.X[:, gene_loc])
        else:
            gene_values = pd.Series(data.raw.X[:, gene_locs].toarray())

        gene_values.index = data.obs_names

        # Marker tSNE plot
        marker_plot_output.clear_output()
        with marker_plot_output:
            py.iplot_mpl(_plot_tsne_markers(data, gene, gene_values), show_link=False)

        # Violin plots
        violin_plot_output.clear_output()
        with violin_plot_output:
             display(_plot_violin_plots(data, gene))

    gene_input.observe(check_gene_input)
    update_button.on_click(update_plots)

    # ------------------------- Cluster Table -------------------------
    cluster_table_title = HTML('<hr><h3>Explore Markers</h3>')

    # Parameters for markers test
    cluster_param_box = widgets.HBox()
    param_c_1 = widgets.Dropdown(options=['cluster', 'rest'] + cluster_names)
    param_c_2 = widgets.Dropdown(options=['cluster', 'rest'] + cluster_names)
    cluster_param_box.children = [param_c_1, param_c_2]

    param_test = widgets.Dropdown(options=['test method', 'wilcoxon', 't-test'], value='test method')
    cluster_table_button = widgets.Button(description='Explore')

    def update_cluster_table(b):
        ident_1 = param_c_1.value
        ident_2 = param_c_2.value
        test = param_test.value

        cluster_table_output.clear_output()
        with cluster_table_output:
            # Validate input
            if (ident_1 == 'cluster') or (ident_2 == 'cluster'):
                display(HTML('Please choose 2 different clusters to compare.'))
                return
            elif ident_1 == ident_2:
                display(HTML('Cannot compare cluster to itself. Choose 2 different clusters to compare.'))
                return
            if test == 'test method':
                display(HTML('Please choose a test method.'))
                return

            table = _find_cluster_markers(data, ident_1, ident_2, test)
            display(table)

    cluster_table_button.on_click(update_cluster_table)

    cluster_table_header_box.children = [cluster_table_title, cluster_param_box, param_test, cluster_table_button]

    # ------------------------- Main Table -------------------------


    # Configure layout
    top_box = widgets.HBox([main_box, sidebar_box])
    return widgets.VBox([top_box, main_table_box])


# def _markers_ui(data, tsne_plots_tab, violin_plots_tab):
#     # Marker Plot UI
#     marker_text_input = widgets.Text(description='Gene List')
#     marker_text_button = widgets.Button(description='Plot')
#     marker_input_box = widgets.HBox([marker_text_input, marker_text_button])
#     header_text = widgets.HTML(
#         '<p>Provide a list of genes (separated by comma) to visualize their expression.</p>')
#     header_box = widgets.VBox([header_text, marker_input_box])
#
#     tsne_plots_tab.layout = Layout(height='800px')
#     tsne_plots_tab.children = [HTML('<p>Plots will display here.</p>')]
#
#     violin_plots_tab.layout = Layout(height='450px')
#     violin_plots_tab.children = [HTML('<p>Plots will display here.</p>')]
#
#     traitlets.link((tsne_plots_tab, 'selected_index'), (violin_plots_tab, 'selected_index'))
#
#     # Create all plots
#     def create_marker_plots(button):
#         # Parse/retrieve gene expression values
#         gene = marker_text_input.value
#         if gene == '':
#             gene_values = [0] * len(data.obs_names)
#         else:
#             gene_list = gene.split(',')
#             gene_list = [x.strip() for x in gene_list]
#             # Handle multiple va
#             if type(gene_list) == list:
#                 gene_locs = [data.raw.var_names.get_loc(x) for x in gene_list]
#             else:
#                 gene_locs = [data.raw.var_names.get_loc[gene_list]]
#
#             if type(data.raw.X) in [np.array, np.ndarray]:
#                 gene_values = pd.DataFrame(data.raw.X[:, gene_locs]).T
#             else:
#                 gene_values = pd.DataFrame(data.raw.X[:, gene_locs].toarray()).T
#
#             gene_values.index = gene_list
#
#         # plots
#         _tsne_markers_ui(data, gene_list, gene_values, tsne_plots_tab, violin_plots_tab)
#
#     marker_text_button.on_click(create_marker_plots)
#
#     table_ui = _cluster_marker_table(data)
#     bottom_box = widgets.HBox([table_ui, violin_plots_tab])
#     return [header_box, tsne_plots_tab, bottom_box]


# def _tsne_markers_ui(data, gene_list, gene_values, tsne_marker_plots_tab, violin_marker_plots_tab):
#     # -------------------- TSNE EXPRESSION PLOT --------------------
#     progress_bar = _create_progress_bar()
#
#     tsne_marker_plots_tab.children = [progress_bar]
#
#     # Calculate and plot marker expression on tSNE plot
#     tsne_figs = _plot_tsne_markers(data, gene_list, gene_values)
#     violin_figs = _plot_violin_plots(data, gene_list, gene_values)
#     tsne_marker_plots_tab.children = [widgets.Output(layout=Layout(
#         display='flex', justify_content='center', align_items='center')) for g in gene_list]
#     violin_marker_plots_tab.children = [widgets.Output(layout=Layout(
#         display='flex', justify_content='center', align_items='center')) for g in gene_list]
#
#     for i, tsne_fig, violin_fig, gene in zip(range(len(tsne_figs)), tsne_figs, violin_figs, gene_list):
#         tsne_marker_plots_tab.set_title(i, gene)
#         violin_marker_plots_tab.set_title(i, gene)
#         with tsne_marker_plots_tab.children[i]:
#             display(HTML('<h4 style="text-align:center;">Relative Expression Across Cells</h4>'))
#             py.iplot_mpl(tsne_fig, show_link=False)
#         with violin_marker_plots_tab.children[i]:
#             display(HTML('<h4 style="text-align:center;">Relative Expression Across Clusters</h4>'))
#             display(violin_fig)
#
#     progress_bar.close()


def _plot_tsne_markers(data, gene, gene_values):
    # Declare grey to red colormap
    expression_cmap = LinearSegmentedColormap.from_list('name', ['lightgrey', 'orangered', 'red'])

    fig = plt.figure(figsize=(5, 5), dpi=100)
    ax = fig.gca()
    sns.regplot(x='tSNE_1', y='tSNE_2', data=pd.DataFrame(data.obsm['X_tsne'], columns=['tSNE_1', 'tSNE_2']), scatter_kws={
                'c': gene_values, 'color': None, 's': 20, 'cmap': expression_cmap}, fit_reg=False, ax=ax)
    ax.set_title(gene)
    plt.tight_layout()
    plt.close()

    return fig


def _plot_violin_plots(data, gene):

    with sns.color_palette('Set2'):
        fig = plt.figure(figsize=(4, 1.8), dpi=100)
        ax = plt.gca()
        sc.pl.violin(data, gene, group_by='louvain_groups', size=2, jitter=1, scale='count', show=False, ax=ax)
        ax.set_xlabel('cluster')
        sns.despine()
        plt.close()

    return fig

# def _cluster_marker_table(data):
#     # -------------------- CLUSTER MARKER TABLE --------------------
#     cell_clusters = data.obs['louvain_groups'].astype(int)
#     cluster_names = np.unique(cell_clusters).tolist()
#     cluster_names = [str(x) for x in cluster_names]
#     num_clusters = len(cluster_names)
#
#     # test used for differential expression analysis
#     test_options = ['wilcoxon', 't-test']
#
#     # -------------------- CUSTOM MARKERS COMPARISON --------------------
#     custom_markers_layout = layout = Layout(width='98%')
#     select_header = widgets.HTML('<p>Choose two clusters to compare markers between.</p>',
#                                  layout=custom_markers_layout)
#     test_dropdown = widgets.Dropdown(options=test_options, value='wilcoxon',
#                                      description='Test', layout=custom_markers_layout)
#     select_1 = widgets.Dropdown(options=cluster_names, value='0',
#                                 description='Cluster 1', layout=custom_markers_layout)
#     select_2 = widgets.Dropdown(options=cluster_names, value='0',
#                                 description='Cluster 2', layout=custom_markers_layout)
#     markers_go_button1 = widgets.Button(description='Calculate Markers')
#     markers_df_output = widgets.Output(layout=Layout(max_height='300px', overflow_y='auto', padding='0'))
#
#     custom_markers_box = widgets.VBox([select_header, test_dropdown, select_1, select_2,
#                                        markers_go_button1, markers_df_output])
#
#     def markers_df_callback(button):
#         markers = _find_cluster_markers(data,
#                                         select_1.value,
#                                         select_2.value,
#                                         test_dropdown.value)
#
#         # Display df
#         markers_df_output.clear_output()
#         with markers_df_output:
#             if type(markers) is pd.DataFrame:
#                 custom_inner_box = widgets.VBox()
#                 custom_markers_output = widgets.Output()
#                 with custom_markers_output:
#                     display(markers)
#                 custom_marker_output_text = HTML(
#                     '<p>Markers positively differentiating cluster {} from cluster {}.</p>'.format(select_1.value, select_2.value))
#                 custom_inner_box.children = [custom_marker_output_text, custom_markers_output]
#                 display(custom_inner_box)
#             else:
#                 display(markers)
#
#     markers_go_button1.on_click(markers_df_callback)
#
#     df_tabs = widgets.Tab([custom_markers_box])
#
#     # -------------------- MARKERS INDIVIDUAL --------------------
#     # Find all markers
#     test_all_dropdown = widgets.Dropdown(
#         options=test_options, value='wilcoxon', description='Test', layout=Layout(max_width='98%'))
#     all_markers_button = widgets.Button(description='Calculate Markers')
#     all_markers_box = widgets.VBox([test_all_dropdown, all_markers_button])
#
#     def all_markers_callback(button=None, tabs=None):
#         marker_df = []
#         df_outputs = {}
#         for cluster in cluster_names:
#             table = _find_cluster_markers(data, cluster, 'rest', test_all_dropdown.value)
#
#             df_outputs[cluster] = widgets.Output(layout=Layout(max_height='300px', overflow_y='auto'))
#             with df_outputs[cluster]:
#                 display(table)
#             marker_df.append(df_outputs[cluster])
#
#         df_tabs.children = marker_df + [custom_markers_box]
#         df_tabs.set_title(num_clusters, 'Custom')
#         for i, c in enumerate(cluster_names):
#             df_tabs.set_title(i, (c))
#
#     all_markers_callback()
#     all_markers_button.on_click(all_markers_callback)
#
#     ui_box = widgets.VBox([all_markers_box, df_tabs], layout=Layout(width='50%'))
#     return ui_box


def _find_cluster_markers(data, ident_1, ident_2, test):
    # Sanitize input for scanpy method
    ident_1 = [str(ident_1)]
    ident_2 = str(ident_2)

    # Perform test
    sc.tl.rank_genes_groups(data, 'louvain_groups', groups=ident_1, reference=ident_2,
                            use_raw=True, n_genes=min(100, len(data.var_names)), test_type=test)

    # Format results
    marker_names = [x[0] for x in data.uns['rank_genes_groups_gene_names']]
    marker_scores = [x[0] for x in data.uns['rank_genes_groups_gene_scores']]

    # Convert to p-values
    marker_scores = st.norm.sf(np.abs(marker_scores))
    marker_scores = multipletests(marker_scores, method='fdr_bh')[1].tolist()
    marker_scores = ['%.3G' % x for x in marker_scores]

    pct_zeroes_1_list = []
    pct_zeroes_2_list = []
    for gene in marker_names:
        clusters = data.obs['louvain_groups'].astype(int)
        is_ident_1 = (clusters == int(ident_1[0]))
        gene_loc = data.raw.var_names.get_loc(gene)
        num_zeroes_1 = data.raw.X[is_ident_1, gene_loc].tolist().count(0)
        num_zeroes_2 = data.raw.X[~is_ident_1, gene_loc].tolist().count(0)
        pct_zeroes_1 = float(num_zeroes_1) / is_ident_1.sum() * 100
        pct_zeroes_2 = float(num_zeroes_2) / (len(is_ident_1) - is_ident_1.sum()) * 100
        pct_zeroes_1_list.append(pct_zeroes_1)
        pct_zeroes_2_list.append(pct_zeroes_2)

    pct_expressed_1 = [100 - e for e in pct_zeroes_1_list]
    pct_expressed_2 = [100 - e for e in pct_zeroes_2_list]

    pct_expressed_1 = ['%.2f' % e for e in pct_expressed_1]
    pct_expressed_2 = ['%.2f' % e for e in pct_expressed_2]



    results = pd.DataFrame([marker_names, marker_scores, pct_expressed_1, pct_expressed_2],
                           index=['Gene', 'Adj p-value', '% expr. {}'.format(ident_1[0]), '% expr. {}'.format(ident_2)]).T
    results.set_index(['Gene'], inplace=True)
    results.index.name = None
    table = TableDisplay(results)

    return table


# -------------------- FILE EXPORT --------------------


def export_data(data, h5ad, path):
    if h5ad:
        if not path.endswith('.h5ad'):
            path = path + '.h5ad'
        data.write(path)

        # User feedback
        display(HTML('Exported data to <code>{}</code> in <code>.h5ad</code> format.'.format(path)))

    else:
        data.write_csvs(path, skip_data=False)

        # User feedback
        display(HTML('Exported data to the <code>{}</code> folder as <code>.csv</code> files.'.format(path)))
