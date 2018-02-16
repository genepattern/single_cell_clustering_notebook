from copy import deepcopy

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as st
import seaborn as sns
from IPython.display import display
from ipywidgets import (HTML, Button, Dropdown, HBox, IntSlider, Layout,
                        Output, SelectionSlider, Tab, Text, VBox)
from matplotlib.ticker import MaxNLocator
from statsmodels.sandbox.stats.multicomp import multipletests

import plotly.offline as py
import plotly.tools as tls
import scanpy.api as sc
from beakerx import TableDisplay, TableDisplayCellHighlighter

py.init_notebook_mode()

# -------------------- HELPERS --------------------

_CLUSTERS_CMAP = 'tab20'
_EXPRESSION_CMAP = LinearSegmentedColormap.from_list(
    'name', ['lightgrey', 'orangered', 'red'])
_LINE_HEIGHT = '20px'


def _create_progress_bar():
    style = '''
        <style>
        @-webkit-keyframes container-rotate {
            to {
                -webkit-transform: rotate(360deg)
            }
        }

        @keyframes container-rotate {
            to {
                -webkit-transform: rotate(360deg);
                transform: rotate(360deg)
            }
        }
        @-webkit-keyframes fill-unfill-rotate {
            12.5% {
                -webkit-transform: rotate(135deg)
            }
            25% {
                -webkit-transform: rotate(270deg)
            }
            37.5% {
                -webkit-transform: rotate(405deg)
            }
            50% {
                -webkit-transform: rotate(540deg)
            }
            62.5% {
                -webkit-transform: rotate(675deg)
            }
            75% {
                -webkit-transform: rotate(810deg)
            }
            87.5% {
                -webkit-transform: rotate(945deg)
            }
            to {
                -webkit-transform: rotate(1080deg)
            }
        }

        @keyframes fill-unfill-rotate {
            12.5% {
                -webkit-transform: rotate(135deg);
                transform: rotate(135deg)
            }
            25% {
                -webkit-transform: rotate(270deg);
                transform: rotate(270deg)
            }
            37.5% {
                -webkit-transform: rotate(405deg);
                transform: rotate(405deg)
            }
            50% {
                -webkit-transform: rotate(540deg);
                transform: rotate(540deg)
            }
            62.5% {
                -webkit-transform: rotate(675deg);
                transform: rotate(675deg)
            }
            75% {
                -webkit-transform: rotate(810deg);
                transform: rotate(810deg)
            }
            87.5% {
                -webkit-transform: rotate(945deg);
                transform: rotate(945deg)
            }
            to {
                -webkit-transform: rotate(1080deg);
                transform: rotate(1080deg)
            }
        }
        .preloader-wrapper.active {
            -webkit-animation: container-rotate 1568ms linear infinite;
            animation: container-rotate 1568ms linear infinite;
        }    .preloader-wrapper {
            display: inline-block;
            position: relative;
            width: 50px;
            height: 50px;
        }
        .spinner-layer {
            position: absolute;
            width: 100%;
            height: 100%;
            opacity: 0;
            border-color: #26a69a;
        }
        .active .spinner-layer {
            border-color: #4285f4;
            opacity: 1;
            -webkit-animation: fill-unfill-rotate 5332ms cubic-bezier(0.4, 0, 0.2, 1) infinite both;
            animation: fill-unfill-rotate 5332ms cubic-bezier(0.4, 0, 0.2, 1) infinite both;
        }
        .circle-clipper {
            display: inline-block;
            position: relative;
            width: 50%;
            height: 100%;
            overflow: hidden;
            border-color: inherit;
        }
        .circle-clipper.left {
            float: left !important;
        }
        .active .circle-clipper.left .circle {
            -webkit-animation: left-spin 1333ms cubic-bezier(0.4, 0, 0.2, 1) infinite both;
            animation: left-spin 1333ms cubic-bezier(0.4, 0, 0.2, 1) infinite both;
        }
        .circle-clipper.left .circle {
            left: 0;
            border-right-color: transparent !important;
            -webkit-transform: rotate(129deg);
            transform: rotate(129deg);
        }
        .circle-clipper .circle {
            width: 200%;
            height: 100%;
            border-width: 3px;
            border-style: solid;
            border-color: inherit;
            border-bottom-color: transparent !important;
            border-radius: 50%;
            -webkit-animation: none;
            animation: none;
            position: absolute;
            top: 0;
            right: 0;
            bottom: 0;
        }
        .circle {
            border-radius: 50%;
        }
        .gap-patch {
            position: absolute;
            top: 0;
            left: 45%;
            width: 10%;
            height: 100%;
            overflow: hidden;
            border-color: inherit;
        }
        .gap-patch .circle {
            width: 1000%;
            left: -450%;
        }
        .circle-clipper {
            display: inline-block;
            position: relative;
            width: 50%;
            height: 100%;
            overflow: hidden;
            border-color: inherit;
        }
        .circle-clipper.right {
            float: right !important;
        }
        .active .circle-clipper.right .circle {
            -webkit-animation: right-spin 1333ms cubic-bezier(0.4, 0, 0.2, 1) infinite both;
            animation: right-spin 1333ms cubic-bezier(0.4, 0, 0.2, 1) infinite both;
        }
        .circle-clipper.right .circle {
            left: -100%;
            border-left-color: transparent !important;
            -webkit-transform: rotate(-129deg);
            transform: rotate(-129deg);
        }
        @-webkit-keyframes left-spin {
            from {
                -webkit-transform: rotate(130deg)
            }
            50% {
                -webkit-transform: rotate(-5deg)
            }
            to {
                -webkit-transform: rotate(130deg)
            }
        }
        @keyframes left-spin {
            from {
                -webkit-transform: rotate(130deg);
                transform: rotate(130deg)
            }
            50% {
                -webkit-transform: rotate(-5deg);
                transform: rotate(-5deg)
            }
            to {
                -webkit-transform: rotate(130deg);
                transform: rotate(130deg)
            }
        }

        @-webkit-keyframes right-spin {
            from {
                -webkit-transform: rotate(-130deg)
            }
            50% {
                -webkit-transform: rotate(5deg)
            }
            to {
                -webkit-transform: rotate(-130deg)
            }
        }

        @keyframes right-spin {
            from {
                -webkit-transform: rotate(-130deg);
                transform: rotate(-130deg)
            }
            50% {
                -webkit-transform: rotate(5deg);
                transform: rotate(5deg)
            }
            to {
                -webkit-transform: rotate(-130deg);
                transform: rotate(-130deg)
            }
        }
    </style>
    '''
    progress_bar = HTML('''{}
    <div class="preloader-wrapper active">
        <div class="spinner-layer spinner-blue-only">
            <div class="circle-clipper left">
                <div class="circle"></div>
            </div><div class="gap-patch">
                <div class="circle"></div>
            </div><div class="circle-clipper right">
                <div class="circle"></div>
            </div>
        </div>
    </div>
    '''.format(style))
    return progress_bar


def _create_placeholder(kind):
    if kind == 'plot':
        word = 'Plot'
    elif kind == 'table':
        word = 'Table'
    placeholder_html = '<p>{} will display here.</p>'.format(word)
    return HTML(placeholder_html, layout=Layout(padding='28px'))


def _create_plot_help():
    return HTML('<b>NOTE:</b> Hover over the plot to interact.')


def _create_export_button(figure, fname):

    # Default png
    filetype_dropdown = Dropdown(
        options=['png', 'svg', 'pdf'],
        value='png',
        layout=Layout(width='75px'))

    # Default filename value
    filename = '../figures/{}.{}'.format(fname, filetype_dropdown.value)
    figure.savefig(
        filename, bbox_inches='tight', format=filetype_dropdown.value)

    # HTML button opens local link on click
    a_style = '''text-decoration: none;
             color: black;
             border: 0px white solid !important;
          '''
    button_style = "width:100px; height:28px; margin:0px;"
    button_classes = "p-Widget jupyter-widgets jupyter-button widget-button"
    button_html = '<button class="{}" style="{}"><a href="{}" target="_blank" style="{}">Save Plot</a>'.format(
        button_classes, button_style, filename, a_style)
    export_button = HTML(button_html)

    # Download file locally
    def save_fig(value_info):
        filename = '../figures/{}.{}'.format(fname, value_info['new'])

        # Disable button until file is properly saved
        export_button.value = '<button class="{}" style="{}" disabled><a href="{}" target="_blank" style="{}">Wait...</a>'.format(
            button_classes, button_style, filename, a_style)

        figure.savefig(filename, bbox_inches='tight', format=value_info['new'])
        export_button.value = '<button class="{}" style="{}"><a href="{}" target="_blank" style="{}">Save Plot</a>'.format(
            button_classes, button_style, filename, a_style)

    filetype_dropdown.observe(save_fig, names='value')

    return HBox(
        [filetype_dropdown, export_button], justify_content='flex-start')


class SingleCellAnalysis:
    """docstring for SingleCellAnalysis."""

    def __init__(self):
        self.data = ''

    # -------------------- SETUP ANALYSIS --------------------
    def setup_analysis(self,
                       matrix_filepath,
                       barcodes_filepath='',
                       genes_filepath=''):
        '''
        Load a raw count matrix for a single-cell RNA-seq experiment.
        Remove lowly expressed genes and cells with low expression.
        '''
        self._setup_analysis(matrix_filepath, barcodes_filepath,
                             genes_filepath)
        self._setup_analysis_ui()

    def _setup_analysis(self, matrix_filepath, barcodes_filepath,
                        genes_filepath):

        # Load data
        if matrix_filepath.endswith('txt') or matrix_filepath.endswith('tsv'):
            data = sc.read_text(matrix_filepath).transpose()
        elif matrix_filepath.endswith('csv'):
            data = sc.read_csv(matrix_filepath).transpose()
        elif matrix_filepath.endswith('mtx'):
            data = sc.read(matrix_filepath, cache=True).transpose()
            data.obs_names = np.genfromtxt(barcodes_filepath, dtype=str)
            data.var_names = np.genfromtxt(genes_filepath, dtype=str)[:, 1]
        else:
            # TODO test
            display(
                HTML('Unknown file type <code>{}</code>.'.format(
                    matrix_filepath.split('.')[-1])))
            return

        # This is needed to setup the "n_genes" column in data.obs.
        sc.pp.filter_cells(data, min_genes=0)

        # Plot some information about mitochondrial genes, important for quality control
        mito_genes = [
            name for name in data.var_names if name.startswith('MT-')
        ]
        data.obs['percent_mito'] = np.sum(
            data[:, mito_genes].X, axis=1) / np.sum(
                data.X, axis=1)
        # add the total counts per cell as observations-annotation to data
        data.obs['n_counts'] = np.sum(data.X, axis=1)
        data.is_log = False
        self.data = data

    def _setup_analysis_ui(self):
        measures = [
            self.data.obs['n_genes'], self.data.obs['n_counts'],
            self.data.obs['percent_mito'] * 100
        ]
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

        fig1_out = Output()
        with fig1_out:
            display(
                _create_export_button(fig1,
                                      '1_setup_analysis_single_qc_plots'))
            display(fig1)

        # Scatter plots of paired variables
        fig2 = plt.figure(2, figsize=(12, 5))
        ax1 = plt.subplot(121)
        ax1 = sns.regplot(
            x='Total Counts',
            y='# of Genes',
            data=pd.DataFrame(
                [measures[1], measures[0]],
                index=['Total Counts', '# of Genes']).T,
            fit_reg=False,
            ax=ax1)
        ax1.set_xlabel(ax1.get_xlabel(), size=14)
        ax1.set_ylabel(ax1.get_ylabel(), size=14)
        ax2 = plt.subplot(122)
        ax2 = sns.regplot(
            x='Total Counts',
            y='% Mitochondrial Genes',
            data=pd.DataFrame(
                [measures[1], measures[2]],
                index=['Total Counts', '% Mitochondrial Genes']).T,
            fit_reg=False,
            ax=ax2)
        ax2.set_xlabel(ax2.get_xlabel(), size=14)
        ax2.set_ylabel(ax2.get_ylabel(), size=14)
        plt.subplots_adjust(wspace=0.3)
        plt.close()

        fig2_out = Output()

        def fig2_out_callback(e):
            fig2_out.clear_output()
            with fig2_out:
                display(_create_plot_help())
                display(
                    _create_export_button(fig2,
                                          '1_setup_analysis_paired_qc_plots'))
                py.iplot_mpl(fig2, show_link=False)

        fig2_out.on_displayed(fig2_out_callback)

        # Descriptive text
        header = HTML(
            '''<div class="alert alert-success" style='font-size:14px; line-height:{};'>
               <p>Loaded <code>{}</code> cells and <code>{}</code> total genes.</p>
               <h3>QC Metrics</h3>
               <p>Use the displayed quality metrics to detect outliers cells and filter unwanted cells below in
               <b>Step 2</b>.
               An abnormally high number of genes or counts in a cell suggests a higher probability of a doublet.
               High levels of mitochondrial genes is characteristic of broken/low quality cells.<br><br>
               Some sensible ranges for this example dataset are:
               <ol>
               <li><code>0 to 2500</code> # of genes per cell</li>
               <li><code>0 to 15000</code> counts per cell</li>
               <li><code>0 to 15%</code> mitochondrial genes per cell</li>
               </p></div>'''
            .format(_LINE_HEIGHT, len(measures[0]), len(self.data.var_names)))
        # Parent container
        tabs = Tab(children=[fig1_out, fig2_out])
        tabs.set_title(0, 'Individual Metrics')
        tabs.set_title(1, 'Pairwise Metrics')
        display(header, tabs)

    # -------------------- PREPROCESS COUNTS --------------------

    def preprocess_counts(self,
                          min_n_cells=0,
                          min_n_genes=0,
                          max_n_genes='inf',
                          min_n_counts=0,
                          max_n_counts='inf',
                          min_percent_mito=0,
                          max_percent_mito='inf',
                          normalization_method='LogNormalize'):
        '''
        Perform cell quality control by evaluating quality metrics, normalize counts, and correct for effects of total counts per cell and the percentage of mitochondrial genes expressed.
        '''
        if min_n_cells == '':
            min_n_cells = 0
        if min_n_genes == '':
            min_n_genes = 0
        if max_n_genes == '':
            max_n_genes = 'inf'
        if min_n_counts == '':
            min_n_counts = 0
        if max_n_counts == '':
            max_n_counts = 'inf'
        if min_percent_mito == '':
            min_percent_mito = 0
        if max_percent_mito == '':
            max_percent_mito = 'inf'

        # Sanitize input
        min_n_cells = float(min_n_cells)
        n_genes_range = [float(min_n_genes), float(max_n_genes)]
        n_counts_range = [float(min_n_counts), float(max_n_counts)]
        percent_mito_range = [float(min_percent_mito), float(max_percent_mito)]

        orig_n_cells = len(self.data.obs_names)
        orig_n_genes = len(self.data.var_names)

        # Perform filtering on genes and cells
        success_run = self._preprocess_counts(
            min_n_cells, n_genes_range, n_counts_range, percent_mito_range,
            normalization_method)
        # Build UI output
        if success_run:
            self._preprocess_counts_ui(orig_n_cells, orig_n_genes)

    def _preprocess_counts(self, min_n_cells, n_genes_range, n_counts_range,
                           percent_mito_range, normalization_method):
        if self.data.raw:
            display(
                HTML(
                    '<div class="alert alert-warning" style="font-size:14px; line-height:{};">This data has already been preprocessed. Please run <code>Step 1: Setup Analysis</code> again if you would like to perform preprocessing again.</div>'.
                    format(_LINE_HEIGHT)))
            return False

        # Gene filtering
        sc.pp.filter_genes(self.data, min_cells=min_n_cells)

        # Filter cells within a range of # of genes and # of counts.
        sc.pp.filter_cells(self.data, min_genes=n_genes_range[0])
        sc.pp.filter_cells(self.data, max_genes=n_genes_range[1])
        sc.pp.filter_cells(self.data, min_counts=n_counts_range[0])
        sc.pp.filter_cells(self.data, max_counts=n_counts_range[1])

        # Remove cells that have too many mitochondrial genes expressed.
        percent_mito_filter = (
            self.data.obs['percent_mito'] * 100 >= percent_mito_range[0]) & (
                self.data.obs['percent_mito'] * 100 < percent_mito_range[1])
        if not percent_mito_filter.any():
            self.data = self.data[percent_mito_filter, :]

        # Set the `.raw` attribute of AnnData object to the logarithmized raw gene expression for later use in
        # differential testing and visualizations of gene expression. This simply freezes the state of the data stored
        # in `data_raw`.
        if normalization_method == 'LogNormalize' and self.data.is_log is False:
            data_raw = sc.pp.log1p(self.data, copy=True)
            self.data.raw = data_raw
        else:
            self.data.raw = deepcopy(self.data.X)

        # Per-cell scaling.
        sc.pp.normalize_per_cell(self.data, counts_per_cell_after=1e4)

        # Identify highly-variable genes.
        sc.pp.filter_genes_dispersion(
            self.data, min_mean=0.1, max_mean=8, min_disp=1)

        # Logarithmize the data.
        if normalization_method == 'LogNormalize' and self.data.is_log is False:
            sc.pp.log1p(self.data)
            self.data.is_log = True

        # Regress out effects of total counts per cell and the percentage of mitochondrial genes expressed.
        sc.pp.regress_out(self.data, ['n_counts', 'percent_mito'])

        # Scale the data to unit variance and zero mean. Clips to max of 10.
        sc.pp.scale(self.data, max_value=10)

        # Calculate PCA
        sc.tl.pca(self.data, n_comps=30)

        # Successfully ran
        return True

    def _preprocess_counts_ui(self, orig_n_cells, orig_n_genes):
        cell_text = '<p><code>{}/{}</code> cells passed filtering.</p>'.format(
            len(self.data.obs_names), orig_n_cells)
        genes_text = '<p><code>{}/{}</code> genes passed filtering.</p>'.format(
            len(self.data.raw.var_names), orig_n_genes)
        v_genes_text = '<p><code>{}/{}</code> genes detected as variable genes.</p>'.format(
            len(self.data.var_names), len(self.data.raw.var_names))
        if self.data.is_log:
            log_text = '<p>Data is log normalized.</p>'
        else:
            log_text = '<p>Data is not normalized.</p>'
        regress_text = '''<p>Performed batch effect removal based on:</p><ol><li># of detected molecules per cell</li>
            <li>% mitochondrial gene content</li></ol>'''
        pca_help_text = '''<h3>Dimensional Reduction: Principal Components</h3>
        <p>Use the following plot showing the standard deviations of the principal components to determine the number of relevant components to use downstream.</p>'''

        output_div = HTML(
            '''<div class='alert alert-success' style='font-size:14px; line-height:{};'>{}{}{}{}{}{}</div>'''.
            format(_LINE_HEIGHT, cell_text, genes_text, v_genes_text, log_text,
                   regress_text, pca_help_text))
        display(output_div)
        display(_create_plot_help())
        pca_fig, pca_py_fig = self._plot_pca()
        display(
            _create_export_button(
                pca_fig, '2_preprocess_counts_pca_variance_ratio_plot'))
        pca_plot_box = Output(layout=Layout(
            display='flex',
            align_items='center',
            justify_content='center',
            margin='0 0 0 -50px'))
        display(pca_plot_box)
        with pca_plot_box:
            py.iplot(pca_py_fig, show_link=False)

    def _plot_pca(self):
        # mpl figure
        fig_elbow_plot = plt.figure(figsize=(6, 5), dpi=100)
        pc_std = self.data.obsm['X_pca'].std(axis=0).tolist()
        pc_std = pd.Series(
            pc_std, index=[x + 1 for x in list(range(len(pc_std)))])
        pc_std = pc_std.iloc[:min(len(pc_std), 30)]
        plt.plot(pc_std, 'o')
        ax = fig_elbow_plot.gca()
        ax.set_xlim(left=0)
        ax.get_xaxis().set_major_locator(MaxNLocator(integer=True))
        ax.get_xaxis().set_minor_locator(MaxNLocator(integer=True))
        ax.set_xlabel('Principal Component', size=16)
        ax.set_ylabel('Variance of PC', size=16)
        plt.close()

        # plot interactive
        py_fig = tls.mpl_to_plotly(fig_elbow_plot)
        py_fig['layout']['margin'] = {'l': 80, 'r': 14, 't': 10, 'b': 45}

        return fig_elbow_plot, py_fig

    # -------------------- Cluster Cells --------------------

    def cluster_cells(self, pcs=10, resolution=1.2, perplexity=30):

        # -------------------- tSNE PLOT --------------------
        pc_sdev = pd.Series(np.std(self.data.obsm['X_pca'], axis=0))
        pc_sdev.index = pc_sdev.index + 1

        # Parameter values
        pc_range = range(2, 31)
        res_range = [
            float('{:0.1f}'.format(x)) for x in list(np.arange(.5, 2.1, 0.1))
        ]
        perp_range = range(5, min(51, len(self.data.obs_names)))

        # Parameter slider widgets
        pc_slider = SelectionSlider(
            options=pc_range,
            value=pcs,
            description="# of PCs",
            continuous_update=False)
        res_slider = SelectionSlider(
            options=res_range,
            value=resolution,
            description="Resolution",
            continuous_update=False)
        perp_slider = SelectionSlider(
            options=perp_range,
            value=perplexity,
            description="Perplexity",
            continuous_update=False)

        # Output widget
        plot_output = Output(layout=Layout(
            height='700px',
            display='flex',
            align_items='center',
            justify_content='center'))
        with plot_output:
            display(_create_placeholder('plot'))

        # "Go" button to plot on click
        def plot_tsne_callback(button=None):
            plot_output.clear_output()
            progress_bar = _create_progress_bar()

            with plot_output:
                # show progress bar
                display(progress_bar)

                # perform tSNE calculation and plot
                self._run_tsne(pc_slider.value, res_slider.value,
                               perp_slider.value)
                tsne_fig, py_tsne_fig = self._plot_tsne(figsize=(8, 6.5))

                display(
                    _create_export_button(
                        tsne_fig, '3_perform_clustering_analysis_tsne_plot'))
                py.iplot(py_tsne_fig, show_link=False)

                # close progress bar
                progress_bar.close()

        # Button widget
        go_button = Button(description='Plot')
        go_button.on_click(plot_tsne_callback)

        # Parameter descriptions
        param_info = HTML(
            '''<div class="alert alert-info" style="font-size:14px;line-height:{};">
            <h3 style="position: relative; top: -10px">Clustering Parameters</h3>
            <p>
            <h4>Number of PCs (Principal Components)</h4>The number of principal components to use in clustering.
            <h4>Resolution</h4>Higher resolution means more and smaller clusters. We find that values 0.6-1.2 typically
            returns good results for single cell datasets of around 3K cells. Optimal resolution often increases for
            larger datasets.
            <h4>Perplexity</h4>The perplexity parameter loosely models the number of close neighbors each point has.
            <a href="https://distill.pub/2016/misread-tsne/">More info on how perplexity matters here</a>.<br><br>
            <b>NOTE:</b> Hover over the plot to interact. Click and drag to zoom. Click on the legend to hide or show
            specific clusters; single-click hides/shows the cluster while double-click isolates the cluster.</p></div>
              '''.format(_LINE_HEIGHT))

        sliders = HBox([pc_slider, res_slider, perp_slider])
        ui = VBox([param_info, sliders, go_button])

        plot_box = VBox([ui, plot_output])
        display(plot_box)
        plot_tsne_callback()

    def _run_tsne(self, pcs, resolution, perplexity):
        sc.tl.tsne(
            self.data,
            n_pcs=pcs,
            perplexity=perplexity,
            learning_rate=1000,
            n_jobs=8)
        sc.tl.louvain(
            self.data,
            n_neighbors=10,
            resolution=resolution,
            recompute_graph=True)

    def _plot_tsne(self, figsize):
        # Clusters
        cell_clusters = self.data.obs['louvain_groups'].astype(int)
        cluster_names = np.unique(cell_clusters).tolist()
        num_clusters = len(cluster_names)

        # Coordinates with cluster assignments
        tsne_coordinates = pd.DataFrame(
            self.data.obsm['X_tsne'],
            index=cell_clusters,
            columns=['tSNE_1', 'tSNE_2'])
        tsne_coordinates['colors'] = cell_clusters.tolist()

        # Cluster color assignments
        palette = sns.color_palette(_CLUSTERS_CMAP, num_clusters)
        colors = dict(zip(cluster_names, palette))

        # Plot each group as a separate trace
        fig, ax = plt.subplots(1, 1, figsize=figsize, dpi=100)
        for c, group in tsne_coordinates.groupby(by='colors'):
            group.plot(
                x='tSNE_1',
                y='tSNE_2',
                kind='scatter',
                c=colors[c],
                label=c,
                alpha=0.7,
                ax=ax,
                legend=False)

        plt.title('tSNE Visualization', size=16)
        ax.set_xlabel(ax.get_xlabel(), size=14)
        ax.set_ylabel(ax.get_ylabel(), size=14)
        plt.tight_layout()
        plt.close()

        py_fig = tls.mpl_to_plotly(fig)
        py_fig['layout']['showlegend'] = True
        py_fig.update(data=[dict(name=c) for c in cluster_names])

        return fig, py_fig

    # -------------------- MARKER ANALYSIS --------------------

    def visualize_markers(self):
        # Commonly used data
        cell_clusters = self.data.obs['louvain_groups'].astype(int)
        cluster_names = np.unique(cell_clusters).tolist()
        cluster_names.sort(key=int)

        # Initialize output widgets here so they are in scope
        marker_plot_tab_1_output = Output(layout=Layout(
            height='600px',
            display='flex',
            justify_content='flex-start',
            align_items='center'))
        marker_plot_tab_2_output = Output(layout=Layout(
            height='600px',
            display='flex',
            justify_content='flex-start',
            align_items='center'))
        violin_plot_output = Output(layout=Layout(
            display='flex',
            justify_content='center',
            align_items='center',
            height='425px',
            width='100%'))
        marker_table_output = Output(layout=Layout(
            display='flex',
            justify_content='flex-start',
            align_items='center',
            height='800px',
            width='100%',
            overflow_y='auto'))
        marker_heatmap_output = Output(layout=Layout(
            display='flex',
            justify_content='center',
            align_items='center',
            height='900px',
            width='100%'))

        # Create main container
        main_box = Tab(layout=Layout(padding='0 12px', width='62%'))

        # t-SNE marker plot container
        main_header_box = VBox()
        marker_plot_tab = Tab(
            [marker_plot_tab_1_output, marker_plot_tab_2_output])
        marker_plot_tab.set_title(0, 'Marker(s) Expression')
        marker_plot_tab.set_title(1, 'Clusters Reference')
        marker_plot_box = VBox(
            [main_header_box, marker_plot_tab, violin_plot_output])

        # Top markers heatmap container
        heatmap_header_box = VBox()
        heatmap_box = VBox([heatmap_header_box, marker_heatmap_output])

        # Populate tabs
        main_box.children = [heatmap_box, marker_plot_box]

        # TODO name tabs
        main_box.set_title(0, 'Heatmap')
        main_box.set_title(1, 'tSNE Plot')
        # Table
        explore_markers_box = VBox(layout=Layout(width='38%'))
        cluster_table_header_box = VBox()
        explore_markers_box.children = [
            cluster_table_header_box, marker_table_output
        ]

        # ------------------------- Output Placeholders -------------------------
        with marker_plot_tab_1_output:
            display(_create_placeholder('plot'))
        with marker_plot_tab_2_output:
            display(_create_placeholder('plot'))

        with violin_plot_output:
            display(_create_placeholder('plot'))
        with marker_table_output:
            display(_create_placeholder('table'))
        with marker_heatmap_output:
            display(_create_placeholder('plot'))

        # Fill container elements
        # ------------------------- Main header -------------------------
        gene_input_description = HTML('''<h3>Visualize Markers</h3>
                                      <p style="font-size:14px; line-height:{};">Visualize the expression of gene(s) in each cell projected on the t-SNE map and the distribution across identified clusters.
                                         Provide any number of genes. If more than one gene is provided, the average expression of those genes will be shown.</p>
                                      '''.format(_LINE_HEIGHT))
        gene_input = Text()
        update_button = Button(description='Plot Expression')
        gene_input_box = HBox([gene_input, update_button])
        main_header_box.children = [
            gene_input_description, _create_plot_help(), gene_input_box
        ]

        def check_gene_input(t):
            '''Don't allow submission of empty input.'''
            if gene_input.value == '':
                update_button.disabled = True
            else:
                update_button.disabled = False

        def update_query_plots(b):
            # Format gene list. Split by comma, remove whitespace, then split by whitespace.
            gene_list = str(gene_input.value).split(',')
            gene_list = [gene.strip().upper() for gene in gene_list]

            if len(gene_list) == 1:
                gene_list = [gene_list[0].strip()]

            # Retrieve expression
            gene_locs = [
                self.data.raw.var_names.get_loc(gene) for gene in gene_list
            ]
            if type(self.data.raw.X) in [np.array, np.ndarray]:
                gene_values = pd.DataFrame(self.data.raw.X[:, gene_locs])
            else:
                gene_values = pd.DataFrame(
                    self.data.raw.X[:, gene_locs].toarray())

            # Final values for plot
            if len(gene_values.shape) > 1:
                values = gene_values.mean(axis=1)
            else:
                values = gene_values
            values.index = self.data.obs_names

            title = ''
            for gene in gene_list:
                if len(title) > 0:
                    title = '{}, {}'.format(title, gene)
                else:
                    title = gene

            # Marker tSNE plot
            tab1_progress_bar = _create_progress_bar()
            marker_plot_tab_1_output.clear_output()
            with marker_plot_tab_1_output:
                display(tab1_progress_bar)

                # generate tSNE markers plot
                tsne_markers_fig = self._plot_tsne_markers(title, values)
                tsne_markers_py_fig = tls.mpl_to_plotly(tsne_markers_fig)

                # Hide progress bar
                tab1_progress_bar.close()

                display(
                    _create_export_button(
                        tsne_markers_fig,
                        '3_perform_clustering_analysis_marker_tsne_plot'))
                py.iplot(tsne_markers_py_fig, show_link=False)

            marker_plot_tab_2_output.clear_output()
            tab2_progress_bar = _create_progress_bar()
            with marker_plot_tab_2_output:
                display(tab2_progress_bar)

                # generate tSNE clusters plot
                tsne_fig, tsne_py_fig = self._plot_tsne(figsize=(5, 5))

                display(
                    _create_export_button(
                        tsne_fig, '3_perform_clustering_analysis_tsne_plot'))
                py.iplot(tsne_py_fig, show_link=False)

                # Hide progress bar
                tab2_progress_bar.close()

            # Violin plots
            violin_plot_output.clear_output()
            with violin_plot_output:
                marker_violin_plot = self._plot_violin_plots(title, values)
                display(
                    _create_export_button(
                        marker_violin_plot,
                        '3_perform_clustering_analysis_marker_violin_plot'))
                display(
                    HTML('<h3>{} Expression Across Clusters</h3>'.format(
                        title)))
                display(marker_violin_plot)

        gene_input.observe(check_gene_input)
        gene_input.on_submit(update_query_plots)
        update_button.on_click(update_query_plots)

        # ------------------------- Heatmap -------------------------

        heatmap_text = HTML(
            '<h3>Visualize Top Markers</h3><p style="font-size:14px; line-height:{};">Show the top markers for each cluster as a heatmap.</p>'.
            format(_LINE_HEIGHT))
        heatmap_n_markers = IntSlider(
            description="# markers", value=10, min=5, max=100, step=5)
        heatmap_test = Dropdown(
            description='test',
            options=['wilcoxon', 't-test'],
            value='wilcoxon')
        heatmap_plot_button = Button(description='Plot')
        heatmap_header_box.children = [
            heatmap_text, heatmap_n_markers, heatmap_test, heatmap_plot_button
        ]

        def plot_heatmap(button=None):
            marker_heatmap_output.clear_output()
            top_marker_progress_bar = _create_progress_bar()
            with marker_heatmap_output:
                display(top_marker_progress_bar)

                expr, group_labels = self._find_top_markers(
                    heatmap_n_markers.value, heatmap_test.value)
                fig = self._plot_top_markers_heatmap(expr, group_labels)

                display(
                    _create_export_button(
                        fig,
                        '3_perform_clustering_analysis_top_markers_heatmap_plot'
                    ))
                display(fig)
                top_marker_progress_bar.close()

        heatmap_plot_button.on_click(plot_heatmap)

        # ------------------------- Cluster Table -------------------------
        cluster_table_header = HTML(
            '<h3>Explore Markers</h3><p style="font-size:14px; line-height:{};">Test for differentially expressed genes between subpopulations of cells.</p>'.
            format(_LINE_HEIGHT))

        # Parameters for markers test
        cluster_param_box = HBox()
        param_c_1 = Dropdown(
            options=['cluster'] + cluster_names,
            value=0,
            layout=Layout(width='90px'))
        param_c_2 = Dropdown(
            options=['cluster', 'rest'] + cluster_names,
            value='rest',
            layout=Layout(width='90px'))
        versus_text = HTML(' vs. ')
        cluster_param_box.children = [param_c_1, versus_text, param_c_2]

        param_test = Dropdown(
            options=['test method', 'wilcoxon', 't-test'],
            value='wilcoxon',
            layout=Layout(width='90px'))
        cluster_table_button = Button(description='Explore')
        cluster_table_note = HTML('''
            <hr>
            <h4>Output Table Info</h4>
            <p style="font-size:14px; line-height:{};">
            <ul style="list-style-position: inside; padding-left: 0; font-size:14px; line-height:{};">
            <li><code>Gene</code>: gene name<br></li>
            <li><code>adj p-value</code>: Benjamini & Hochberg procedure adjusted p-values<br></li>
            <li><code>avg logFC</code>: log fold-change of average relative expression of gene in the first group compared to the second group<br></li>
            <li><code>pct.#</code>: number of cells in the first group that express the gene<br></li>
            <li><code>pct.#</code>: number of cells in the second group that express the gene<br></li>
            </ul>
            <hr>
            <b>Note:</b> Export the table using the menu, which can be accessed in the top left hand corner of the "Gene" column.</p>'''
                                  .format(_LINE_HEIGHT, _LINE_HEIGHT))

        def update_cluster_table(b):
            ident_1 = param_c_1.value
            ident_2 = param_c_2.value
            test = param_test.value

            marker_table_output.clear_output()
            marker_table_progress_bar = _create_progress_bar()
            with marker_table_output:
                # Validate input
                if (ident_1 == 'cluster') or (ident_2 == 'cluster'):
                    display(
                        HTML('Please choose 2 different clusters to compare.'))
                    return
                elif ident_1 == ident_2:
                    display(
                        HTML(
                            'Cannot compare cluster to itself. Choose 2 different clusters to compare.'
                        ))
                    return
                if test == 'test method':
                    display(HTML('Please choose a test method.'))
                    return

                display(marker_table_progress_bar)

                # Find markers for specified
                table = self._find_markers(ident_1, ident_2, test)
                display(table)
                marker_table_progress_bar.close()

        cluster_table_button.on_click(update_cluster_table)

        cluster_table_header_box.children = [
            cluster_table_header, cluster_table_note, cluster_param_box,
            param_test, cluster_table_button
        ]

        # ------------------------- Main Table -------------------------

        # Configure layout
        top_box = HBox([main_box, explore_markers_box])
        display(top_box)
        plot_heatmap()

    def _plot_tsne_markers(self, title, gene_values):
        fig, ax = plt.subplots(1, 1, figsize=(5, 5), dpi=100)
        sns.regplot(
            x='tSNE_1',
            y='tSNE_2',
            data=pd.DataFrame(
                self.data.obsm['X_tsne'], columns=['tSNE_1', 'tSNE_2']),
            scatter_kws={
                'c': gene_values,
                'color': None,
                's': 20,
                'cmap': _EXPRESSION_CMAP
            },
            fit_reg=False,
            ax=ax)
        ax.set_title(title)
        plt.tight_layout()
        plt.close()

        return fig

    def _plot_violin_plots(self, gene, gene_values):
        fig = plt.figure(figsize=(5.5, 3), dpi=100)
        ax = plt.gca()
        groups = self.data.obs['louvain_groups']
        sns.stripplot(
            x=groups, y=gene_values, size=3, jitter=True, color='black', ax=ax)
        sns.violinplot(
            x=groups,
            y=gene_values,
            hue=groups,
            width=0.5,
            dodge=False,
            palette=_CLUSTERS_CMAP,
            ax=ax)
        plt.legend(loc=(1, 0.1))
        ax.set_xlabel('cluster')
        sns.despine()
        plt.close()

        return fig

    def _find_markers(self, ident_1, ident_2, test):
        # Sanitize input for scanpy method
        ident_1 = str(ident_1)
        ident_2 = str(ident_2)

        # Perform test
        sc.tl.rank_genes_groups(
            self.data,
            'louvain_groups',
            groups=[ident_1],
            reference=ident_2,
            use_raw=True,
            n_genes=min(100, len(self.data.var_names)),
            test_type=test)

        # Format results
        marker_names = [
            x[0] for x in self.data.uns['rank_genes_groups_gene_names']
        ]
        marker_scores = [
            x[0] for x in self.data.uns['rank_genes_groups_gene_scores']
        ]

        # Convert to p-values
        marker_scores = st.norm.sf(np.abs(marker_scores))
        marker_scores = multipletests(
            marker_scores, method='fdr_bh')[1].tolist()
        marker_scores = ['%.3G' % x for x in marker_scores]

        clusters = self.data.obs['louvain_groups'].astype(int)
        is_ident_1 = (clusters == int(ident_1))
        if ident_2 is not 'rest':
            is_ident_2 = (clusters == int(ident_2))

        # gene_locs = [self.data.raw.var_names.get_loc(gene) for gene in marker_names]
        if type(self.data.raw.X) not in [np.array, np.ndarray]:
            df = pd.DataFrame(
                self.data.raw.X.toarray(),
                index=self.data.obs_names,
                columns=self.data.raw.var_names)
        else:
            df = pd.DataFrame(
                self.data.raw.X,
                index=self.data.obs_names,
                columns=self.data.raw.var_names)

        # Determine mean of each gene across each group
        mean_1 = np.mean(df.loc[is_ident_1, marker_names])
        if ident_2 == 'rest':
            mean_2 = np.mean(df.loc[~is_ident_1, marker_names])
        else:
            mean_2 = np.mean(df.loc[is_ident_2, marker_names])

        # Compute log fold change for each gene
        log_fc = list(mean_1 / mean_2)
        log_fc = ['%.2f' % v for v in log_fc]
        log_fc = [float(x) for x in log_fc]
        # Replace 'inf' with 0
        log_fc = [0 if e == float('inf') else e for e in log_fc]

        # Compute percent expressed in each group
        pct_1 = (df.loc[is_ident_1, marker_names] > 0
                 ).sum() / is_ident_1.sum() * 100
        if ident_2 == 'rest':
            pct_2 = (df.loc[~is_ident_1, marker_names] > 0).sum() / (
                len(self.data.obs_names) - is_ident_1.sum()) * 100
        else:
            pct_2 = (df.loc[is_ident_2, marker_names] > 0
                     ).sum() / is_ident_2.sum() * 100

        # Format to 2 decimal places
        pct_1 = ['%.2f' % e for e in pct_1]
        pct_2 = ['%.2f' % e for e in pct_2]

        # Return as interactive table
        results = pd.DataFrame(
            [marker_names, marker_scores, log_fc, pct_1, pct_2],
            index=[
                'Gene', 'adj p-value', 'avg logFC', 'pct.{}'.format(ident_1),
                'pct.{}'.format(ident_2)
            ]).T
        results.set_index(['Gene'], inplace=True)
        table = TableDisplay(results)
        for c in results.columns:
            # flip for p-value
            if c == results.columns[0]:
                highlighter = TableDisplayCellHighlighter.getHeatmapHighlighter(
                    c,
                    TableDisplayCellHighlighter.SINGLE_COLUMN,
                    minColor='red',
                    maxColor='grey')
            else:
                highlighter = TableDisplayCellHighlighter.getHeatmapHighlighter(
                    c,
                    TableDisplayCellHighlighter.SINGLE_COLUMN,
                    minColor='grey',
                    maxColor='red')

            table.addCellHighlighter(highlighter)

        return table

    def _find_top_markers(self, n_markers, test):
        sc.tl.rank_genes_groups(
            self.data,
            'louvain_groups',
            use_raw=True,
            n_genes=min(n_markers, len(self.data.var_names)),
            test_type=test)

        # genes sorted by top (rows), clusters (columns)
        markers_per_cluster = pd.DataFrame(
            self.data.uns['rank_genes_groups_gene_names'])
        markers = np.array([
            markers_per_cluster[c].values.tolist()
            for c in markers_per_cluster.columns
        ]).flatten()
        marker_locs = [self.data.raw.var_names.get_loc(m) for m in markers]

        # clusters
        clusters = self.data.obs['louvain_groups'].astype(int)
        cluster_names = clusters.unique().tolist()
        cluster_names.sort(key=int)

        # get expression for markers
        expr = self.data.raw.X[:, marker_locs]
        if type(expr) not in [np.array, np.ndarray]:
            expr = expr.toarray()

        # format dataframe
        expr = expr.transpose()
        expr = pd.DataFrame(
            expr,
            index=self.data.raw.var_names[marker_locs],
            columns=self.data.obs_names)

        cells_order = []
        group_labels = []
        df_grouped = expr.groupby(clusters, axis=1)

        for g in cluster_names:
            cluster_df = df_grouped.get_group(g)
            cells_order.extend(cluster_df.columns.tolist())
            group_labels.extend([g] * len(cluster_df.columns))

        expr = expr.loc[:, cells_order]
        return expr, group_labels

    def _plot_top_markers_heatmap(self, counts, group_labels):
        clusters = self.data.obs['louvain_groups'].astype(int)
        cluster_names = clusters.unique().tolist()
        cluster_names.sort(key=int)

        num_clusters = len(cluster_names)
        num_markers = int(len(counts.index) / len(cluster_names))

        # Row/column cluster color annotation
        cmap = dict(
            zip(
                np.unique(group_labels),
                sns.color_palette(_CLUSTERS_CMAP, n_colors=num_clusters)))
        cell_colors = pd.Series(group_labels, index=counts.columns).map(cmap)
        gene_labels = np.array([[c] * num_markers
                                for c in cluster_names]).flatten()
        gene_colors = pd.Series(gene_labels).map(cmap).tolist()

        g = sns.clustermap(
            counts,
            row_cluster=False,
            col_cluster=False,
            row_colors=gene_colors,
            col_colors=cell_colors,
            xticklabels=False,
            figsize=(8, 14),
            cmap=_EXPRESSION_CMAP)

        hm = g.ax_heatmap
        hm.set_yticklabels(counts.index, {'fontsize': '10'})
        hm.set_yticks([x + 0.5 for x in range(len(counts.index))])

        plt.close()
        return g.fig

    # -------------------- FILE EXPORT --------------------

    def export_data(self, path, h5ad=False):
        if h5ad:
            if not path.endswith('.h5ad'):
                path = path + '.h5ad'
            self.data.write(path)

            # User feedback
            display(
                HTML('''
                <div class="alert alert-success">
                <p style="font-size:14px; line-height:{};">Exported data to <code>{}</code> in <code>.h5ad</code> format.</p>
                </div>'''.format(_LINE_HEIGHT, path)))

        else:
            self.data.write_csvs(path, skip_data=False)

            # User feedback
            display(
                HTML('''
                <div class="alert alert-success">
                <p style="font-size:14px; line-height:{};">Exported data to the <a href="{}" target="_blank">{}</a> folder as <code>.csv</code> files.</p>
                </div>'''.format(_LINE_HEIGHT, path, path)))
