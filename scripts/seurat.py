import warnings

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
import rpy2.robjects as robjects
from plotly import graph_objs as go
from rpy2.rinterface import RRuntimeWarning
from rpy2.robjects import pandas2ri
from rpy2.robjects.packages import importr

py.init_notebook_mode()

pandas2ri.activate()


class SeuratExperiment:

    _experiment = None
    _n_genes = None
    _n_umi = None
    _percent_mito = None
    _num_cells = None
    _num_genes = None
    _scale_num_cells = None
    _num_var_genes = None
    _pc_sdev = None
    _tsne_coordinates = None
    _cell_clusters = None
    _num_clusters = None
    _cluster_names = None

    _CONFIG = {'showLink': False}

    def __init__(self):
        sns.set_style("whitegrid")
        warnings.filterwarnings("ignore", category=RRuntimeWarning)
        importr('Seurat')
        robjects.r('suppressWarnings(library("dplyr"))')
        importr('Matrix')

    # -------------------- HELPERs, SETTERS, GETTERS --------------------
    def slot(self, robject, slotname):
        return robjects.r['slot'](robject, slotname)

    def get_raw_data(self):
        df = pandas2ri.ri2py_dataframe(robjects.r['as.matrix'](self.slot(self._experiment, 'raw.data')))
        df.index = robjects.r['rownames'](self.slot(self._experiment, 'raw.data'))
        df.columns = robjects.r['colnames'](self.slot(self._experiment, 'raw.data'))
        return df

    def get_scale_data(self):
        df = pandas2ri.ri2py_dataframe(robjects.r['as.matrix'](self.slot(self._experiment, 'scale.data')))
        df.index = robjects.r['rownames'](self.slot(self._experiment, 'scale.data'))
        df.columns = robjects.r['colnames'](self.slot(self._experiment, 'scale.data'))
        return df
    # -------------------- LOAD DATASET --------------------

    def load_dataset(self, path, min_cells=3, min_genes=200):
        self._load_dataset(path, min_cells, min_genes)
        self._load_dataset_ui()

    def _load_dataset(self, path, min_cells, min_genes):
        rstring = '''
      function(path, min_cells, min_genes) {
        data <- read.table(path, row.names = 1)
        data = Matrix(as.matrix(data), sparse = TRUE)
        e <- CreateSeuratObject(data, min.cells = min_cells, min.genes = min_genes)
        mito.genes <- grep(pattern = "^MT-", x = rownames(x = e@data), value = TRUE)
        percent.mito <- Matrix::colSums(e@raw.data[mito.genes, ]) / Matrix::colSums(e@raw.data)
        e <- AddMetaData(object = e, metadata = percent.mito, col.name = "percent.mito")
        return(list(e, e@meta.data$nGene, e@meta.data$nUMI, percent.mito))
      }
      '''
        # Create R Seurat Object
        function = robjects.r(rstring)
        experiment, n_genes, n_umi, percent_mito = function(
            path, min_cells, min_genes)

        # Save data
        self._experiment = experiment
        self._n_genes = pandas2ri.ri2py(n_genes)
        self._n_umi = pandas2ri.ri2py(n_umi)
        self._percent_mito = pandas2ri.ri2py(percent_mito)
        self._num_genes, self._num_cells = self.get_raw_data().shape

    def _load_dataset_ui(self):
        measures = [self._n_genes, self._n_umi, self._percent_mito]
        measure_names = ['# of Genes', '# of UMI', '% Mitochondrial Genes']

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
        ax1 = sns.regplot(x='# of UMI', y='# of Genes', data=pd.DataFrame(
            [self._n_umi, self._n_genes], index=['# of UMI', '# of Genes']).T, fit_reg=False, ax=ax1)
        ax1.set_xlabel(ax1.get_xlabel(), size=14)
        ax1.set_ylabel(ax1.get_ylabel(), size=14)
        ax2 = plt.subplot(122)
        ax2 = sns.regplot(x='# of UMI', y='% Mitochondrial Genes', data=pd.DataFrame(
            [self._n_umi, self._percent_mito], index=['# of UMI', '% Mitochondrial Genes']).T, fit_reg=False, ax=ax2)
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
               <p>*For non-UMI data, the # of UMI represents the sum of the non-normalized values within a cell.</p>
               <p>We can identify and exclude outlier cells based on some quality control metrics.</p>'''.format(self._num_cells, self._num_genes))
        # Parent container
        tabs = widgets.Tab(children=[fig1_out, fig2_out])
        tabs.set_title(0, 'Single Variables')
        tabs.set_title(1, 'Pairwise Variables')
        display(header, tabs)

    def preprocess_counts(self, n_genes_range=['-inf', 'inf'], percent_mito_range=['-inf', 'inf'], normalization_method='LogNormalize'):
        self._preprocess_counts(n_genes_range, percent_mito_range, normalization_method)
        self._preprocess_counts_ui()

    def _preprocess_counts(self, n_genes_range, percent_mito_range, normalization_method):
        rstring = '''
            function(experiment, n_genes_range, percent_mito_range, normalization_method) {
            n_genes_range <- as.numeric(n_genes_range)
            percent_mito_range <- as.numeric(percent_mito_range)
            low.thresholds <- c(n_genes_range[1], percent_mito_range[1])
            high.thresholds <- c(n_genes_range[2], percent_mito_range[2])
            experiment <- FilterCells(object = experiment, subset.names = c("nGene", "percent.mito"), low.thresholds = low.thresholds, high.thresholds = high.thresholds)

            experiment <- NormalizeData(object = experiment, normalization.method = normalization_method, scale.factor = 1e4, display.progress=FALSE)
            experiment <- FindVariableGenes(object = experiment, do.plot = FALSE, display.progress = FALSE)
            experiment <- ScaleData(object = experiment, vars.to.regress = c("nUMI", "percent.mito"), display.progress = FALSE)

            return(experiment)
        }
        '''

        # Format input
        if type(n_genes_range) is str:
            n_genes_range = n_genes_range.split(',')

        if type(percent_mito_range) is str:
            percent_mito_range = percent_mito_range.split(',')

        # Format R output for python
        function = robjects.r(rstring)
        self._experiment = function(self._experiment, n_genes_range,
                                    percent_mito_range, normalization_method)

        self._scale_num_cells = self.get_scale_data().shape[1]
        self._num_var_genes = pandas2ri.ri2py(robjects.r['length'](self.slot(self._experiment, 'var.genes')))[0]

    def _preprocess_counts_ui(self):
        cell_text = HTML(
            '<p><code>{}/{}</code> cells passed filtering.</p>'.format(self._scale_num_cells, self._num_cells))
        v_genes_text = HTML('<p><code>{}</code> variable genes detected.</p>'.format(self._num_var_genes))
        regress_text = HTML(
            '''<p>Performed batch effect removal based on:</p><ol><li># of detected molecules per cell</li>
            <li>% mitochondrial gene content.</li></ol>''')
        display(cell_text, v_genes_text, regress_text)

    def perform_clustering_analysis(self):
        self._run_pca()

        elbow_output = widgets.Output()
        pcs = 10
        resolution = 0.9
        perplexity = 13

        def plot_output_callback(e):
            # mpl figure
            fig_elbow_plot = plt.figure(figsize=(5, 4))
            plt.plot(self._pc_sdev, 'o')
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
                py.iplot_mpl(fig_elbow_plot, show_link=False)

        # -------------------- tSNE PLOT --------------------
        # Parameter values
        pc_range = range(2, len(self._pc_sdev) + 1)
        res_range = [float('{:0.1f}'.format(x))
                     for x in list(np.arange(.1, 3.1, 0.1))]
        perp_range = range(5, min(100, self._scale_num_cells))

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
                self._run_tsne(pc_slider.value, res_slider.value, perp_slider.value)
                self._cluster_names = list(np.unique(self._cell_clusters.ident.values))

                py_fig = self._plot_tsne()
                py.iplot(py_fig, show_link=False)

            self._markers_ui(markers_ui_output)

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

    def _run_pca(self):
        rstring = '''
          function(experiment) {
            pcs <- min(dim(experiment@scale.data)[2], 30)
            experiment <- RunPCA(object = experiment, pcs.compute = pcs, do.print = FALSE)
            experiment <- ProjectPCA(object = experiment, pcs.store = pcs, do.print = FALSE)

            return(list(experiment, GetDimReduction(experiment, slot = 'sdev')))
          }
        '''
        function = robjects.r(rstring)
        self._experiment, pc_sdev = function(self._experiment)
        self._pc_sdev = pandas2ri.ri2py(pc_sdev)

    def _run_tsne(self, pcs, resolution, perplexity):
        rstring = '''
          function(experiment, pcs, resolution, perplexity) {
            experiment <- FindClusters(object = experiment, reduction.type = "pca", dims.use = 1:pcs, resolution = resolution, print.output = 0, save.SNN = TRUE)
            experiment <- RunTSNE(object = experiment, dims.use = 1:pcs, do.fast = TRUE, perplexity = perplexity)

            # Return coordinates and cluster identities
            tsne.coord <- as.data.frame(experiment@dr$tsne@cell.embeddings)
            cell.ident <- as.data.frame(experiment@ident)

            return(list(experiment, tsne.coord, cell.ident))
          }
        '''
        # Output tSNE coordinates and cluster annotations
        function = robjects.r(rstring)
        self._experiment, tsne_coordinates, cell_clusters = function(self._experiment, pcs, resolution, perplexity)
        self._tsne_coordinates = pandas2ri.ri2py(tsne_coordinates)
        self._cell_clusters = pandas2ri.ri2py(cell_clusters)
        self._cell_clusters.columns = ['ident']
        self._num_clusters = len(pd.unique(self._cell_clusters.values.flatten()))

    def _plot_tsne(self):
        palette = sns.color_palette("Set2", self._num_clusters)
        colors = dict(zip(self._cluster_names, palette))
        cluster_colors = self._cell_clusters.ident.map(dict(zip(self._cluster_names, palette))).tolist()

        df = self._tsne_coordinates
        df['colors'] = self._cell_clusters.ident

        fig, ax = plt.subplots(1, 1, figsize=(8, 8))
        for c, group in df.groupby(by='colors'):
            group.plot(x='tSNE_1', y='tSNE_2', kind='scatter', c=colors[c], label=c, ax=ax, legend=False)
            # ax = sns.regplot(x='tSNE_1', y='tSNE_2', data=group, color=colors[c], fit_reg=False, label=c)
        # ax.legend(loc='best')
        # ax = sns.regplot(x='tSNE_1', y='tSNE_2', data=self._tsne_coordinates,
        #                  scatter_kws={'color': cluster_colors}, fit_reg=False)
        plt.title('tSNE Visualization', size=16)
        ax.set_xlabel(ax.get_xlabel(), size=14)
        ax.set_ylabel(ax.get_ylabel(), size=14)
        plt.close()

        py_fig = tls.mpl_to_plotly(fig)
        py_fig['layout']['showlegend'] = True
        py_fig.update(data=[dict(name=c) for c in self._cluster_names])

        return py_fig

    def _markers_ui(self, markers_ui_output):
        markers_plot_output = widgets.Output()

        # -------------------- CLUSTER MARKER TABLE --------------------
        markers_ui_output.clear_output()
        with markers_ui_output:

            # test used for differential expression analysis
            test_options = ['bimod', 'wilcox', 'roc', 't',
                            'tobit', 'poisson', 'negbinom', 'MAST', 'DESeq2']
            test_dropdown = widgets.Dropdown(
                options=test_options, value='bimod', description='Test to use')

            # -------------------- CUSTOM COMPARISON --------------------
            # selection groups
            select_1_header = widgets.HTML('<h4>Group 1</h4>')
            select_2_header = widgets.HTML('<h4>Group 2</h4>')
            select_1 = widgets.SelectMultiple(options=self._cluster_names, value=[
                                              '0'], layout=Layout(max_width='100%', margin='2px'))
            select_2 = widgets.SelectMultiple(options=self._cluster_names, value=[
                                              _ for _ in self._cluster_names if _ != '0'], layout=Layout(max_width='100%', margin='2px'))
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
                markers = self._find_cluster_markers(
                    list(select_1.value), list(select_2.value), test_dropdown.value)

                # Display df
                markers_df_output.clear_output()
                with markers_df_output:
                    display(markers)

            markers_go_button1.on_click(markers_df_callback)

            df_tabs = widgets.Tab([custom_markers_box])

            # -------------------- MARKERS INDIVIDUAL --------------------
            # Find all markers
            test_all_dropdown = widgets.Dropdown(
                options=test_options, value='bimod', description='Test to use')
            all_markers_button = widgets.Button(description='Recalculate Markers')
            all_markers_box = widgets.VBox([test_all_dropdown, all_markers_button])

            def all_markers_callback(button=None, tabs=None):
                marker_df = []
                df_outputs = {}
                for cluster in self._cluster_names:
                    df = self._find_cluster_markers(
                        cluster, [c for c in self._cluster_names if c != cluster], test_all_dropdown.value)
                    df_outputs[cluster] = widgets.Output(layout=Layout(max_height='400px', overflow_y='scroll'))

                    with df_outputs[cluster]:
                        display(df)

                    marker_df.append(df_outputs[cluster])

                df_tabs.children = marker_df + [custom_markers_box]
                df_tabs.set_title(self._num_clusters, 'Custom')
                for i, c in enumerate(self._cluster_names):
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
                fig = self._plot_tsne_markers(gene)
                markers_plot_output.clear_output()
                with markers_plot_output:
                    # display(fig)
                    py.iplot_mpl(fig, show_link=False)

            marker_text_button.on_click(plot_marker)

            header = widgets.HTML('<p>Provide up to 9 genes separated by comma to visualize the expression of each gene.</p>')
            plot_markers_box = widgets.VBox([header, marker_input_box, markers_plot_output], layout=Layout(width='62%'))

            markers_ui_box = widgets.VBox([all_markers_box, df_tabs], layout=Layout(width='38%'))
            markers_all_box = widgets.HBox([markers_ui_box, plot_markers_box])
            display(markers_all_box)

        return markers_ui_output

    def _find_all_cluster_markers(self, test, top_num=50):
        rstring = '''
        function(experiment, test, top_num) {
            markers <- FindAllMarkers(object = experiment,
                              test.use = test,
                              min.pct = 0.25,
                              only.pos = TRUE,
                              print.bar = FALSE)
            return(markers %>% group_by(cluster) %>% top_n(top_num, avg_logFC))
        }
        '''
        function = robjects.r(rstring)
        markers = function(self._experiment, test, top_num)
        markers = pandas2ri.ri2py(markers)
        markers.index = markers.gene
        markers = markers.loc[:, markers.columns != 'gene']
        markers = markers.loc[:, markers.columns != 'p_val']

        # Formatting dataframe headers
        markers.index.name = 'Gene'
        return markers

    def _find_cluster_markers(self, ident_1, ident_2, test):
        rstring = '''
        function(experiment, ident_1, ident_2, test) {
            return(FindMarkers(object = experiment,
                            ident.1 = as.vector(ident_1),
                            ident.2 = as.vector(ident_2),
                            test.use = test,
                            min.pct = 0.25,
                            only.pos = TRUE,
                            print.bar = FALSE))
        }
        '''
        function = robjects.r(rstring)
        markers = function(self._experiment, ident_1, ident_2, test)
        markers = pandas2ri.ri2py(markers)
        markers = markers.loc[:, markers.columns != 'p_val']

        # Formatting dataframe headers
        markers.index.name = 'Gene'
        return markers

    def _plot_marker_expression(self, gene):
        df = pd.DataFrame([self._cell_clusters.ident, self.get_scale_data().loc[gene]]).T
        df.columns = ['g', 'x']
        df = df.astype({'g': 'category', 'x': 'float'})
        # Initialize the FacetGrid object
        cluster_names = list(np.unique(self._cell_clusters.ident.values))
        pal = sns.cubehelix_palette(len(cluster_names), rot=3, light=.8, dark=0.25, hue=1.5)
        sns.violinplot(x='g', y='x', hue='g', data=df)
        sns.despine()
        fig = plt.gcf()
        plt.close()
        return fig

    def _plot_tsne_markers(self, gene=''):
        # Retrieve gene expression values
        if gene == '':
            gene_values = [0] * self.get_scale_data().shape[1]
        else:
            gene_list = gene.split(',')
            gene_list = [x.strip() for x in gene_list]
            if len(gene_list) > 1:
                gene_values = self.get_scale_data().loc[gene_list]
            else:
                gene_values = self.get_scale_data().loc[gene_list]

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

            sns.regplot(x='tSNE_1', y='tSNE_2', data=self._tsne_coordinates, scatter_kws={
                        'c': gene_values.loc[g], 'color': None, 's': marker_size, 'cmap': expression_cmap}, fit_reg=False, ax=ax)
            ax.set_title(g, y=0.92)
            ax.set_ylabel('')
            ax.set_xlabel('')
        plt.subplots_adjust(hspace=0.3)
        plt.close()

        # fig = plt.figure(figsize=(8, 8))
        # ax = sns.regplot(x='tSNE_1', y='tSNE_2', data=self._tsne_coordinates, scatter_kws={'c': gene_values, 'color': None, 'cmap' : expression_cmap}, fit_reg=False)
        # # cbar = ColorbarBase(ax, cmap=expression_cmap, orientation='horizontal')
        #
        # plt.title('{} Expression'.format(gene), size=16)
        # ax.set_xlabel(ax.get_xlabel(), size=14)
        # ax.set_ylabel(ax.get_ylabel(), size=14)
        # plt.close()
        return fig
