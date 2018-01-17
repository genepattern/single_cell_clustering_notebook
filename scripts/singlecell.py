import ipywidgets as widgets
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from ipywidgets import HTML, Layout
from matplotlib.colorbar import ColorbarBase
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import MaxNLocator

import plotly.offline as py
import plotly.tools as tls
import scanpy.api as sc

py.init_notebook_mode()

# -------------------- LOAD DATASET --------------------


def load_dataset(path, min_cells=3, min_genes=200):
    data = _load_dataset(path, min_cells, min_genes)
    _load_dataset_ui(data)
    return data


def _load_dataset(path, min_cells, min_genes):

    # Load data
    data = sc.read_text(path).transpose()

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
    fig2 = plt.figure(2, figsize=(12, 5))
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


def preprocess_counts(data, n_genes_range=['-inf', 'inf'], percent_mito_range=['-inf', 'inf'], normalization_method='LogNormalize'):
    if type(n_genes_range) is str:
        n_genes_range = n_genes_range.split(',')

    if type(percent_mito_range) is str:
        percent_mito_range = percent_mito_range.split(',')

    n_genes_range = [float(x) for x in n_genes_range]
    percent_mito_range = [float(x) for x in percent_mito_range]

    data, data_raw = _preprocess_counts(data, n_genes_range, percent_mito_range, normalization_method)
    _preprocess_counts_ui(data, data_raw)
    return data


def _preprocess_counts(data, n_genes_range, percent_mito_range, normalization_method):
    if data.raw:
        display(HTML('This data has already been preprocessed. Please run <code>load_dataset()</code> to reprocess this data.'))
        return data, data.raw.X

    # Remove cells that have too many mitochondrial genes expressed or too many total counts.
    # Actually do the filtering.
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

    data_raw = sc.pp.log1p(data, copy=True)
    data.raw = data_raw

    # Per-cell normalize the data matrix $\mathbf{X}$, identify highly-variable genes and compute logarithm.
    sc.pp.normalize_per_cell(data, counts_per_cell_after=1e4)
    filter_result = sc.pp.filter_genes_dispersion(
        data.X, min_mean=0.1, max_mean=8, min_disp=1)

    # Actually do the filtering.
    data = data[:, filter_result.gene_subset]

    # Logarithmize the data.
    sc.pp.log1p(data)

    # Regress out effects of total counts per cell and the percentage of
    # mitochondrial genes expressed. Scale the data to unit variance.
    sc.pp.regress_out(data, ['n_counts', 'percent_mito'])
    sc.pp.scale(data, max_value=10)

    # Save the result.
    # data.write(results_file)
    return data, data_raw


def _preprocess_counts_ui(data, data_raw):
    cell_text = HTML(
        '<p><code>{}/{}</code> cells passed filtering.</p>'.format(len(data.obs_names), data_raw.shape[0]))
    v_genes_text = HTML('<p><code>{}</code> variable genes detected.</p>'.format(len(data.var_names)))
    regress_text = HTML(
        '''<p>Performed batch effect removal based on:</p><ol><li># of detected molecules per cell</li>
        <li>% mitochondrial gene content.</li></ol>''')
    display(cell_text, v_genes_text, regress_text)

# -------------------- PERFORM CLUSTERING & MARKER ANALYSIS --------------------


def perform_clustering_analysis(data):
    data = _run_pca(data)
    pc_sdev = pd.Series(np.std(data.obsm['X_pca'], axis=0))
    pc_sdev.index = pc_sdev.index + 1

    elbow_output = widgets.Output()
    pcs = 10
    resolution = 0.9
    perplexity = 13

    def plot_output_callback(e):
        # mpl figure
        fig_elbow_plot = plt.figure(figsize=(4.5, 5))
        plt.plot(pc_sdev, 'o')
        ax = fig_elbow_plot.gca()
        x_axis = ax.axes.get_xaxis()
        y_axis = ax.axes.get_yaxis()
        x_axis.set_major_locator(MaxNLocator(integer=True))
        y_axis.set_major_locator(MaxNLocator(integer=True))
        ax.set_xlabel('PC', size=16)
        ax.set_ylabel('Standard Deviation of PC', size=14)
        plt.close()

        # plot
        with elbow_output:
            py_fig = tls.mpl_to_plotly(fig_elbow_plot)
            py_fig['layout']['margin'] = {'l': 35, 'r': 14, 't': 10, 'b': 45}
            py.iplot(py_fig, show_link=False)

    # -------------------- tSNE PLOT --------------------
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

    plot_layout = Layout(width='60%')
    side_layout = Layout(width='40%')

    # Output widget
    plot_output = widgets.Output(layout=plot_layout)
    markers_ui_output = widgets.Output()

    # "Go" button to plot on click
    def plot_tsne_callback(button):
        # show tSNE plot
        plot_output.clear_output()
        with plot_output:
            _run_tsne(data, pc_slider.value, res_slider.value, perp_slider.value)

            py_fig = _plot_tsne(data)
            py.iplot(py_fig, show_link=False)

        _markers_ui(data, markers_ui_output)

    # Button widget
    go_button = widgets.Button(description='Plot')
    go_button.on_click(plot_tsne_callback)

    # Parameter descriptions
    pc_info = HTML('<p>The number of selected principal components to use in clustering. Determine the number of principal components (PCs) to use by drawing a cutoff where there is a clear elbow in the graph.</p>')
    pc_box = widgets.VBox([pc_info, elbow_output])
    res_info = HTML('<p>The resolution parameter sets the ‘granularity’ of the downstream clustering, with increased values leading to a greater number of clusters. We find that setting this parameter between 0.6-1.2 typically returns good results for single cell datasets of around 3K cells. Optimal resolution often increases for larger datasets.</p>')
    perp_info = HTML('<p>The perpelexity parameter loosely models the number of close neighbors each point has. More info on how perplexity matters <a href="https://distill.pub/2016/misread-tsne/">here</a>.</p>')

    param_header = HTML('<h4>Parameters</h4>')

    param_info = widgets.Accordion(
        [pc_box, res_info, perp_info])
    param_info.set_title(0, 'Number of PCs (Principal Components)')
    param_info.set_title(1, 'Resolution')
    param_info.set_title(2, 'Perplexity')

    sliders = widgets.VBox([pc_slider, res_slider, perp_slider])

    ui = widgets.VBox([param_header, sliders, go_button, param_info], layout=side_layout)

    plot_box = widgets.HBox([ui, plot_output])
    plot_box.on_displayed(plot_output_callback)
    tabs = widgets.Tab([plot_box, markers_ui_output])

    tabs.set_title(0, '1. Visualize Clusters')
    tabs.set_title(1, '2. Explore Markers')
    display(tabs)

    return data


def _run_pca(data):
    sc.tl.pca(data)
    data.obsm['X_pca'] *= -1  # multiply by -1 to match Seurat R
    return data


def _run_tsne(data, pcs, resolution, perplexity):
    sc.tl.tsne(data, n_pcs=pcs, perplexity=perplexity, learning_rate=200, n_jobs=8)
    sc.tl.louvain(data, n_neighbors=40, resolution=resolution, recompute_graph=True)

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

    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    for c, group in df.groupby(by='colors'):
        group.plot(x='tSNE_1', y='tSNE_2', kind='scatter', c=colors[c], label=c, ax=ax, legend=False)
        # ax = sns.regplot(x='tSNE_1', y='tSNE_2', data=group, color=colors[c], fit_reg=False, label=c)
    # ax.legend(loc='best')
    # ax = sns.regplot(x='tSNE_1', y='tSNE_2', data=tsne_coordinates,
    #                  scatter_kws={'color': cluster_colors}, fit_reg=False)
    plt.title('tSNE Visualization', size=16)
    ax.set_xlabel(ax.get_xlabel(), size=14)
    ax.set_ylabel(ax.get_ylabel(), size=14)
    plt.close()

    py_fig = tls.mpl_to_plotly(fig)
    py_fig['layout']['showlegend'] = True
    py_fig.update(data=[dict(name=c) for c in cluster_names])

    return py_fig


def _markers_ui(data, markers_ui_output):
    markers_plot_output = widgets.Output()

    # -------------------- CLUSTER MARKER TABLE --------------------
    markers_ui_output.clear_output()
    with markers_ui_output:

        cell_clusters = data.obs['louvain_groups'].astype(int)
        cluster_names = np.unique(cell_clusters).tolist()
        cluster_names = [str(x) for x in cluster_names]
        num_clusters = len(cluster_names)

        # test used for differential expression analysis
        test_options = ['wilcoxon', 't-test']
        test_dropdown = widgets.Dropdown(
            options=test_options, value='wilcoxon', description='Test to use')

        # -------------------- CUSTOM COMPARISON --------------------
        # selection groups
        select_1_header = widgets.HTML('<h4>Group 1</h4>')
        select_2_header = widgets.HTML('<h4>Group 2</h4>')
        select_1 = widgets.SelectMultiple(options=cluster_names, value=[
                                          '0'], layout=Layout(max_width='100%', margin='2px'))
        select_2 = widgets.SelectMultiple(options=cluster_names, value=[
                                          _ for _ in cluster_names if _ != '0'], layout=Layout(max_width='100%', margin='2px'))
        select_1_box = widgets.VBox([select_1_header, select_1], layout=Layout(max_width='50%', margin='8px'))
        select_2_box = widgets.VBox([select_2_header, select_2], layout=Layout(max_width='50%', margin='8px'))
        selection_box = widgets.HBox([select_1_box, select_2_box])

        # Go button
        markers_go_button1 = widgets.Button(description='Calculate Markers')
        markers_df_output = widgets.Output(layout=Layout(max_height='400px', overflow_y='scroll'))

        select_header = widgets.HTML(
            '<p>Highlight clusters to show markers for Group 1 vs. Group 2. Multiple clusters can be selected with shift and/or ctrl (or command) pressed and mouse clicks or arrow keys.</p>')
        custom_markers_box = widgets.VBox([select_header, test_dropdown, selection_box,
                                           markers_go_button1, markers_df_output])

        def markers_df_callback(button):
            markers = _find_cluster_markers(data,
                                            list(select_1.value),
                                            list(select_2.value),
                                            test_dropdown.value)

            # Display df
            markers_df_output.clear_output()
            with markers_df_output:
                display(markers)

        markers_go_button1.on_click(markers_df_callback)

        df_tabs = widgets.Tab([custom_markers_box])

        # -------------------- MARKERS INDIVIDUAL --------------------
        # Find all markers
        test_all_dropdown = widgets.Dropdown(
            options=test_options, value='wilcoxon', description='Test to use')
        all_markers_button = widgets.Button(description='Recalculate Markers')
        all_markers_box = widgets.VBox([test_all_dropdown, all_markers_button])

        def all_markers_callback(button=None, tabs=None):
            marker_df = []
            df_outputs = {}
            for cluster in cluster_names:
                df = _find_cluster_markers(data,
                                           cluster,
                                           [c for c in cluster_names if c != cluster],
                                           test_all_dropdown.value)
                df_outputs[cluster] = widgets.Output(layout=Layout(max_height='400px', overflow_y='scroll'))

                with df_outputs[cluster]:
                    display(df)

                marker_df.append(df_outputs[cluster])

            df_tabs.children = marker_df + [custom_markers_box]
            df_tabs.set_title(num_clusters, 'Custom')
            for i, c in enumerate(cluster_names):
                df_tabs.set_title(i, c)

        all_markers_callback()
        all_markers_button.on_click(all_markers_callback)

        # -------------------- MARKERS PLOT --------------------

        # Marker Plot UI
        marker_text_input = widgets.Text(description='Gene')
        marker_text_button = widgets.Button(description='Plot')
        marker_input_box = widgets.HBox([marker_text_input, marker_text_button])

        def plot_marker(e):
            gene = marker_text_input.value
            fig = _plot_tsne_markers(data, gene)
            markers_plot_output.clear_output()
            with markers_plot_output:
                py.iplot_mpl(fig, show_link=False)

        marker_text_button.on_click(plot_marker)

        header = widgets.HTML(
            '<p>Provide up to 9 genes separated by comma to visualize the expression of each gene.</p>')
        plot_markers_box = widgets.VBox(
            [header, marker_input_box, markers_plot_output], layout=Layout(width='62%'))

        markers_ui_box = widgets.VBox([all_markers_box, df_tabs], layout=Layout(width='38%'))
        markers_all_box = widgets.HBox([markers_ui_box, plot_markers_box])
        display(markers_all_box)

    return markers_ui_output


def _find_cluster_markers(data, ident_1, ident_2, test):
    sc.tl.rank_genes_groups(data, 'louvain_groups', groups=ident_1, reference='rest',
                            use_raw=False, n_genes=min(100, len(data.var_names)), test_type=test)
    return pd.DataFrame(data.uns['rank_genes_groups_gene_names'])


def _plot_tsne_markers(data, gene=''):
    # Retrieve gene expression values
    if gene == '':
        gene_values = [0] * len(data.obs_names)
    else:
        gene_list = gene.split(',')
        gene_list = [x.strip() for x in gene_list]

        if type(gene_list) == list:
            gene_locs = [data.var_names.get_loc(x) for x in gene_list]
        else:
            gene_locs = [data.var_names.get_loc[gene_list]]

        gene_values = pd.DataFrame(data.X[:, gene_locs]).T
        gene_values.index = gene_list

    # Declare grey to red colormap
    expression_cmap = LinearSegmentedColormap.from_list('name', ['lightgrey', 'red'])

    # Plot each gene
    if len(gene_list) == 1:
        fig, axes = plt.subplots(1, 1, figsize=(8, 8), sharex=True, sharey=True)
        marker_size = 20
    elif len(gene_list) <= 4:
        fig, axes = plt.subplots(2, 2, figsize=(8, 8), sharex=True, sharey=True)
        marker_size = 10
    else:
        fig, axes = plt.subplots(3, 3, figsize=(8, 8), sharex=True, sharey=True)
        marker_size = 4

    for i, g in enumerate(gene_values.index):
        if len(gene_list) == 1:
            ax = axes
        elif len(gene_list) <= 4:
            ax_x = int(i / 2)
            ax_y = i % 2
            ax = axes[ax_x][ax_y]
        else:
            ax_x = int(i / 3)
            ax_y = i % 3
            ax = axes[ax_x][ax_y]
        # plt.scatter(x=data.obsm['X_tsne'][:,0], y=data.obsm['X_tsne'][:,1], c=gene_values.loc[g])
        sns.regplot(x='tSNE_1', y='tSNE_2', data=pd.DataFrame(data.obsm['X_tsne'], columns=['tSNE_1', 'tSNE_2']), scatter_kws={
                    'c': gene_values.loc[g], 'color': None, 's': marker_size, 'cmap': expression_cmap}, fit_reg=False, ax=ax)
        ax.set_title(g, y=0.92)
        ax.set_ylabel('')
        ax.set_xlabel('')
    plt.subplots_adjust(hspace=0.3)
    plt.close()

    # fig = plt.figure(figsize=(8, 8))
    # ax = sns.regplot(x='tSNE_1', y='tSNE_2', data=_tsne_coordinates, scatter_kws={'c': gene_values, 'color': None, 'cmap' : expression_cmap}, fit_reg=False)
    # # cbar = ColorbarBase(ax, cmap=expression_cmap, orientation='horizontal')
    #
    # plt.title('{} Expression'.format(gene), size=16)
    # ax.set_xlabel(ax.get_xlabel(), size=14)
    # ax.set_ylabel(ax.get_ylabel(), size=14)
    # plt.close()
    return fig

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
