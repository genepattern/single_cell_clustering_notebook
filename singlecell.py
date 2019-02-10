import os
import warnings
from copy import deepcopy

import matplotlib as mpl
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import scipy.stats as st
import seaborn as sns
from IPython.display import display
from ipywidgets import (HTML, Accordion, Button, Dropdown, FloatProgress,
                        FloatRangeSlider, HBox, IntRangeSlider, IntSlider,
                        Layout, Output, SelectionSlider, Tab, Text, VBox,
                        interactive_output)
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import MaxNLocator
from statsmodels.sandbox.stats.multicomp import multipletests
import subprocess

import plotly.offline as py
import plotly.tools as tls
import scanpy.api as sc
from beakerx import TableDisplay, TableDisplayCellHighlighter

import qgrid

py.init_notebook_mode()
warnings.simplefilter('ignore', UserWarning)

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
    progress_bar = HTML(
        '''{}
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
    '''.format(style),
        layout=Layout(height='150px', width='100%'))
    return progress_bar


def _create_placeholder(kind):
    if kind == 'plot':
        word = 'Plot'
    elif kind == 'table':
        word = 'Table'
    placeholder_html = '<p>{} will display here.</p>'.format(word)
    return HTML(placeholder_html, layout=Layout(padding='28px'))


def _info_message(message):
    return HTML(
        '<div class="alert alert-info" style="font-size:14px; line-height:20px;"><p><b>NOTE:</b> {}</p></div>'.
        format(message))


def _output_message(message):
    return HTML(
        '<div class="well well-sm" style="font-size:14px; line-height:20px; padding: 15px;">{}</div>'.
        format(message))

def _output_message_txt(message):
    return '<div class="well well-sm" style="font-size:14px; line-height:20px; padding: 15px;">{}</div>'.format(message)

def _warning_message(message):
    return HTML(
        '<div class="alert alert-warning" style="font-size:14px; line-height:20px;">{}</div>'.
        format(message))

def _error_message(message):
    return HTML(
        '<div class="alert alert-danger" style="font-size:14px; line-height:20px;">{}</div>'.
        format(message))

"""
These functions generate and update HTML objects to
display notebook progress status
"""

def _get_new_status(message):
    return _output_message('<h3>Progress: </h3>'+message)

def _update_status(stat, message):
    stat.value = _output_message_txt('<h3>Progress: </h3>'+message)

def _create_export_button(figure, fname):
    # Default png
    filetype_dropdown = Dropdown(
        options=['png', 'svg', 'pdf'],
        value='png',
        layout=Layout(width='75px'))

    if not os.path.isdir('figures'):
        os.mkdir('figures')

    # Default filename value
    filename = 'figures/{}.{}'.format(fname, filetype_dropdown.value)
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
        if not os.path.isdir('figures'):
            os.mkdir('figures')
        filename = 'figures/{}.{}'.format(fname, value_info['new'])

        # Disable button until file is properly saved
        export_button.value = '<button class="{}" style="{}" disabled><a href="{}" target="_blank" style="{}">Wait...</a>'.format(
            button_classes, button_style, filename, a_style)

        figure.savefig(filename, bbox_inches='tight', format=value_info['new'])
        export_button.value = '<button class="{}" style="{}"><a href="{}" target="_blank" style="{}">Save Plot</a>'.format(
            button_classes, button_style, filename, a_style)

    filetype_dropdown.observe(save_fig, names='value')

    return HBox(
        [filetype_dropdown, export_button], justify_content='flex-start')


def _download_text_file(url):
    '''
    Downloads file assuming simple text file. Returns name of file.
    '''
    filename = url.split('/')[-1]

    r = requests.get(url, stream=True)
    file_size = int(r.headers['Content-Length'])
    chunk_size = int(file_size / 50)
    if chunk_size == 0:
        chunk_size = int(file_size)

    progress_bar = FloatProgress(
        value=0,
        min=0,
        max=50,
        step=1,
        description='Loading...',
        bar_style='info',
        orientation='horizontal')

    display(progress_bar)
    f = open(filename, 'wb')
    for chunk in r.iter_content(chunk_size=chunk_size):
        f.write(chunk)
        progress_bar.value += 1

    f.close()
    progress_bar.close()
    #display(HTML('<p>Downloaded file: <code>{}</code>.</p>'.format(filename)))

    return filename

class SingleCellAnalysis:
    """docstring for SingleCellAnalysis."""

    def __init__(self, verbose=False):
        self.data = ''
        self.verbose = verbose
        mpl.rcParams['figure.dpi'] = 80

    # -------------------- SETUP ANALYSIS --------------------
    def setup_analysis(self, csv_filepath=None, gene_x_cell=True, mtx_filepath=None, 
                        gene_filepath=None, bc_filepath=None):
        '''
        Load a raw count matrix for a single-cell RNA-seq experiment.

        If data is a single matrix file, csv_filepath should be used
        and the other three variable names should be None. If data is in
        10x format (mtx, gene, barcode), csv_filepath should be None
        and the other three variables should be used correspondingly
        '''
        # Hide FutureWarnings.
        warnings.simplefilter('ignore',
                              FutureWarning) if self.verbose else None

        stat = _get_new_status("Preparing your files...")
        display(stat)

        if not self._setup_analysis(csv_filepath, gene_x_cell, mtx_filepath, 
                                    gene_filepath, bc_filepath, stat):
            return

        _update_status(stat, "Building QC metric plots...")

        self._setup_analysis_ui()

        _update_status(stat, "Done! QC for your data is displayed below")

        # Revert to default settings to show FutureWarnings.
        warnings.simplefilter('default',
                              FutureWarning) if self.verbose else None

    def _setup_analysis(self, csv_filepath, gene_x_cell, mtx_filepath, gene_filepath,
                        bc_filepath, stat):
        # Check for either one matrix file or all populated 10x fields, make
        # sure user did not choose both or neither
        paths_10x = [mtx_filepath, gene_filepath, bc_filepath]
        use_csv = (csv_filepath != [])
        use_10x = (paths_10x != [[], [], []])
        if use_csv and use_10x:
            display(_error_message("Can't use both single matrix file and 10X"))
            return False
        elif not use_csv and not use_10x:
            display(_error_message("Must supply single matrix file or 10X files"))
            return False

        if use_csv:
            local_csv_filepath = csv_filepath

            if local_csv_filepath.startswith('http'):
                _update_status(stat, "Downloading "+local_csv_filepath+"...")
                display(stat)
                local_csv_filepath = _download_text_file(csv_filepath)

            if local_csv_filepath.endswith('.zip'):
                subprocess.call('unzip -o '+local_csv_filepath, shell=True)
                local_csv_filepath = '.'.join(local_csv_filepath.split('.')[:-1])

            data = sc.read(local_csv_filepath, cache=False)
            if gene_x_cell:
                data = data.transpose()

        elif use_10x:
            local_mtx_filepath = mtx_filepath
            local_gene_filepath = gene_filepath
            local_bc_filepath = bc_filepath

            if mtx_filepath.startswith('http'):
                _update_status(stat, "Downloading "+mtx_filepath+"...")
                local_mtx_filepath = _download_text_file(mtx_filepath)

            if gene_filepath.startswith('http'):
                _update_status(stat, "Downloading "+gene_filepath+"...")
                local_gene_filepath = _download_text_file(gene_filepath)

            if bc_filepath.startswith('http'):
                _update_status(stat, "Downloading "+bc_filepath+"...")
                local_bc_filepath = _download_text_file(bc_filepath)

            if local_mtx_filepath.endswith('.zip'):
                _update_status(stat, "Unpacking "+local_mtx_filepath+"...")
                subprocess.call('unzip -o '+local_mtx_filepath, shell=True)
                local_mtx_filepath = '.'.join(local_mtx_filepath.split('.')[:-1])

            if local_gene_filepath.endswith('.zip'):
                _update_status(stat, "Unpacking "+local_gene_filepath+"...")
                subprocess.call('unzip -o '+local_gene_filepath, shell=True)
                local_gene_filepath = '.'.join(local_gene_filepath.split('.')[:-1])

            if local_bc_filepath.endswith('.zip'):
                _update_status(stat, "Unpacking "+local_bc_filepath+"...")
                subprocess.call('unzip -o '+local_bc_filepath, shell=True)
                local_bc_filepath = '.'.join(local_bc_filepath.split('.')[:-1]) 

            _update_status(stat, "Loading "+local_mtx_filepath+"...")
            data = sc.read(local_mtx_filepath, cache=False).transpose()

            _update_status(stat, "Loading "+local_bc_filepath+"...")
            data.obs_names = np.genfromtxt(local_bc_filepath, dtype=str)

            _update_status(stat, "Loading "+local_gene_filepath+"...")
            data.var_names = np.genfromtxt(local_gene_filepath, dtype=str)[:,1]

        # This is needed to setup the "n_genes" column in data.obs.
        sc.pp.filter_cells(data, min_genes=0)

        # Plot some information about mitochondrial genes, important for quality control
        _update_status(stat, "Calculating mitochondrial DNA QC metrics...")
        mito_genes = [
            name for name in data.var_names if name.startswith('MT-')
        ]
        data.obs['percent_mito'] = np.sum(
            data[:, mito_genes].X, axis=1) / np.sum(
                data.X, axis=1)

        # add the total counts per cell as observations-annotation to data
        _update_status(stat, "Calculating gene counts QC metrics...")
        data.obs['n_counts'] = np.sum(data.X, axis=1)
        data.is_log = False
        self.data = data
        return True

    def _setup_analysis_ui(self):
        measures = pd.DataFrame([
            self.data.obs['n_genes'], self.data.obs['n_counts'],
            self.data.obs['percent_mito'] * 100
        ]).T

        measure_names = {
            'n_genes': '# of Genes',
            'n_counts': 'Total Counts',
            'percent_mito': '% Mitochondrial Genes'
        }

        def plot_fig1(a, b, c):
            with sns.axes_style('ticks'):
                fig1 = plt.figure(figsize=(14, 4))
                gs = mpl.gridspec.GridSpec(2, 3, wspace=0.1, hspace=0.05, height_ratios=[20, 1])

                selected_info = HBox(layout=Layout( width='812px', margin='0 0 0 10px'))
                selected_info_children = []
                is_selected = [True] * measures.shape[0]

                for measure, ax_col, color, w in zip(measures.columns, list(range(len(measures.columns))), sns.color_palette('Set1')[:3], [a, b, c]):
                    values = measures[measure]
                    # Draw density plots
                    ax = plt.subplot(gs[0, ax_col])
                    if sum(values) > 0:
                        sns.kdeplot(values, shade=True, ax=ax, color=color, legend=False)
                    ax.set_xlim(0)
                    plt.setp(ax.get_xticklabels(), visible=False)
                    plt.setp(ax.get_yticklabels(), visible=False)
                    plt.setp(ax.get_yticklines(), visible=False)
                    ax.set_xlabel('')
                    sns.despine(ax=ax)

                    # Draw mean
                    ax.plot(len(ax.get_ylim()) * [values.mean()], ax.get_ylim(), color=color)
                    ax.text(values.mean(), ax.get_ylim()[1], 'x̅', fontsize=13)

                    # Draw SD lines
                    for i in range(3, 5):
                        ycoords = list(ax.get_ylim())
                        ycoords[1] *= (1-0.16 * (i-2))
                        if values.mean()+(i*values.std()) > values.max():
                            continue
                        ax.plot(len(ax.get_ylim()) * [values.mean() + i * values.std()],
                                ycoords, linestyle=':', color=color)
                        ax.text(values.mean() + i * values.std(), ax.get_ylim()
                                [1] * (1-0.16 * (i-2)), 'x̅ + {}σ'.format(i), fontsize=13)

                    # Draw selected area
                    selected_area = patches.Rectangle((w[0], 0), w[1], len(ax.get_ylim()),
                                                      linewidth=0, facecolor='black', alpha=0.1)
                    ax.add_patch(selected_area)

                    # Calculate # of cells selected using all filters
                    is_selected = is_selected & (measures[measure] >= w[0]) & (measures[measure] <= w[1])

                    # Draw points on bottom
                    ax_1 = plt.subplot(gs[1, ax_col], sharex=ax)
                    sns.stripplot(values, ax=ax_1, color=color, orient='horizontal', size=3, jitter=True, alpha=0.3)
                    ax_1.set_xlim(0)
                    ax_1.set_xlabel(measure_names[ax_1.get_xlabel()])
                    plt.setp(ax_1.get_xticklines(), visible=False)
                    plt.setp(ax_1.get_yticklines(), visible=False)
                    sns.despine(ax=ax_1, left=True, bottom=True)

                    # Update selected info
                    w_info = HTML(value='<font size=4>Range: <code>{:.2f} - {:.2f}</code></font>'.format(w[0], w[1]), 
                                  layout=Layout(width='270px', padding='0', margin='0 0 0 25px'))
                    selected_info_children.append(w_info)

            plt.close()
            selected_info.children = selected_info_children
            is_selected_info = HTML('<font size=4><code><b>{:.2f}% ({} / {})</b></code> of total cells will be selected.</font>'.format(sum(is_selected) /
                                                                                                            len(is_selected) * 100, 
                                                                                                            sum(is_selected), 
                                                                                                            len(is_selected)), 
                                    layout=Layout(margin='0 0 0 200px'))
            display(
                _create_export_button(fig1,
                                      '1_setup_analysis_single_qc_plots'), is_selected_info, fig1, selected_info)

        slider_box = HBox(layout=Layout(width='812px', margin='0 0 0 0'))
        slider_box_children = []
        for measure in measures:
            values = measures[measure]
            slider = FloatRangeSlider(value=[0, values.mean() + 3 * values.std()],
                                      min=0,
                                      max=values.max(),
                                      step=0.01,
                                      continuous_update=False,
                                      readout=False,
                                      layout=Layout(margin='0 20px 0 40px'))
            slider_box_children.append(slider)

        slider_box.children = slider_box_children

        slider_box_children.append(slider)
        fig1_out = Output()
        with fig1_out:
            interactive_fig1 = interactive_output(plot_fig1, dict(zip(['a', 'b', 'c'], slider_box.children)))
            interactive_fig1.layout.height = '450px'
            display(interactive_fig1, slider_box)

        # Descriptive text
        header = _output_message('''<h3>Results</h3>
        <p>Loaded <code>{}</code> cells and <code>{}</code> total genes.</p>
        <h3>QC Metrics</h3>
        <p>Visually inspect the quality metric distributions to visually identify thresholds for
        filtering unwanted cell. Filtering is performed in
        <b>Step 2</b>.<br><br>
        There are 3 metrics including:
        <ol>
        <li>the number of genes detected in each cell</li>
        <li>the total read counts in each cell</li>
        <li>the percentage of counts mapped to mitochondrial genes</li>
        </ol>
        A high percentage of reads mapped to mitochondrial genes indicates the cell may have lysed before isolation,
        losing cytoplasmic RNA and retaining RNA enclosed in the mitochondria. An abnormally high number of genes
        or counts in a cell suggests a higher probability of a doublet.
        <br><br>
        A standard upper threshold for removing outliers is roughly <i>3-4 standard deviations</i> above the mean.
        Inspect the quality metric distribution plots below to filter appropriately.
        </p>'''.format(measures.shape[0], len(self.data.var_names)))

        display(header, fig1_out)

    # -------------------- PREPROCESS COUNTS --------------------

    def preprocess_counts(self,
                          min_n_cells=0,
                          min_n_genes=0,
                          max_n_genes='inf',
                          min_n_counts=0,
                          max_n_counts='inf',
                          min_percent_mito=0,
                          max_percent_mito='inf',
                          normalization_method='LogNormalize',
                          do_regression=True):
        '''
        Perform cell quality control by evaluating quality metrics, normalizing counts, scaling, and correcting for effects of total counts per cell and the percentage of mitochondrial genes expressed. Also detect highly variable genes and perform linear dimensional reduction (PCA).
        '''
        # Hide FutureWarnings.
        stat = _get_new_status("Preparing to preprocess data...")
        display(stat)

        warnings.simplefilter('ignore',
                              FutureWarning) if self.verbose else None

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
            normalization_method, stat, do_regression)

        # Build UI output
        if success_run:
            _update_status(stat, "Preparing preprocessing results visualizations...")
            self._preprocess_counts_ui(orig_n_cells, orig_n_genes)
            _update_status(stat, "Done! See the results of preprocessing below")

        # Revert to default settings to show FutureWarnings.
        warnings.simplefilter('default',
                              FutureWarning) if self.verbose else None

    def _preprocess_counts(self, min_n_cells, n_genes_range, n_counts_range,
                           percent_mito_range, normalization_method, stat,
                           do_regression):
        if self.data.raw:
            display(
                _warning_message(
                    'This data has already been preprocessed. Please run <a href="#Step-1:-Setup-Analysis">Step 1: Setup Analysis</a> again if you would like to perform preprocessing again.</div>'
                ))
            return False

        _update_status(stat, "Filtering cells by #genes and #counts...")


        # Gene filtering
        sc.pp.filter_genes(self.data, min_cells=min_n_cells)



        # Filter cells within a range of # of genes and # of counts.
        sc.pp.filter_cells(self.data, min_genes=n_genes_range[0])
        sc.pp.filter_cells(self.data, max_genes=n_genes_range[1])
        sc.pp.filter_cells(self.data, min_counts=n_counts_range[0])
        sc.pp.filter_cells(self.data, max_counts=n_counts_range[1])


        # Remove cells that have too many mitochondrial genes expressed.
        _update_status(stat, "Removing cells high in mitochondrial genes...")
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

        # Per-cell scaling.
        _update_status(stat, "Performing per-cell normalization...")
        sc.pp.normalize_per_cell(self.data, counts_per_cell_after=1e4)

        # Identify highly-variable genes.
        _update_status(stat, "Identifying highly-variable genes...")
        sc.pp.filter_genes_dispersion(
            self.data, min_mean=0.0125, max_mean=3, min_disp=0.5)

        # Logarithmize the data.
        if normalization_method == 'LogNormalize' and self.data.is_log is False:
            _update_status(stat, "Log-normalizing data...")
            sc.pp.log1p(self.data)
            self.data.is_log = True

        # Regress out effects of total counts per cell and the percentage of mitochondrial genes expressed.
        if do_regression:
            _update_status(stat, "Performing regression based on counts per cell and percent mitochondrial genes expressed...")
            sc.pp.regress_out(self.data, ['n_counts', 'percent_mito'])

        # Scale the data to unit variance and zero mean. Clips to max of 10.
        _update_status(stat, "Scaling data to have unit variance and zero mean...")
        sc.pp.scale(self.data, max_value=10)

        # Calculate PCA
        _update_status(stat, "Performing principle component analysis (PCA)...")
        sc.tl.pca(self.data, n_comps=30)

        # Successfully ran
        return True

    def _preprocess_counts_ui(self, orig_n_cells, orig_n_genes):
        cell_text = '<p>Number of cells passed filtering: <code>{} / {}</code></p>'.format(
            len(self.data.obs_names), orig_n_cells)
        genes_text = '<p>Number of genes passed filtering: <code>{} / {}</code></p>'.format(
            len(self.data.raw.var_names), orig_n_genes)
        v_genes_text = '<p>Number of genes detected as variable genes: <code>{} / {}</code></p>'.format(
            len(self.data.var_names), len(self.data.raw.var_names))
        if self.data.is_log:
            log_text = '<p>Data is log normalized.</p>'
        else:
            log_text = '<p>Data is not normalized.</p>'
        regress_text = '''<p>Performed linear regression to remove unwanted sources of variation including:<br>
                          <ol><li># of detected molecules per cell</li>
                              <li>% mitochondrial gene content</li>
                          </ol></p>'''
        pca_help_text = '''<h3>Dimensional Reduction: Principal Components</h3>
        <p>Use the following plot showing the standard deviations of the principal components to determine the number of relevant components to use downstream.</p>'''

        output_div = _output_message(
            '''<h3 style="position: relative; top: -10px">Preprocess Counts Results</h3>{}{}{}{}{}{}'''.
            format(cell_text, genes_text, v_genes_text, log_text, regress_text,
                   pca_help_text))
        display(output_div)
        display(_info_message('Hover over the plot to interact.'))
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
        fig_elbow_plot = plt.figure(figsize=(7, 6))
        pc_var = self.data.uns['pca_variance_ratio']
        pc_var = pc_var[:min(len(pc_var), 30)]

        # Calculate percent variance explained
        pc_var = [v / sum(pc_var) * 100 for v in pc_var]

        pc_var = pd.Series(pc_var, index=[x + 1 for x in range(len(pc_var))])

        plt.plot(pc_var, 'o')
        ax = fig_elbow_plot.gca()
        ax.set_xlim(left=0)
        ax.get_xaxis().set_major_locator(MaxNLocator(integer=True))
        ax.get_xaxis().set_minor_locator(MaxNLocator(integer=True))
        ax.set_xlabel('Principal Component', size=16)
        ax.set_ylabel('% Variance Explained', size=16)
        plt.close()

        # plot interactive
        py_fig = tls.mpl_to_plotly(fig_elbow_plot)
        py_fig['layout']['margin'] = {'l': 75, 'r': 14, 't': 10, 'b': 45}

        return fig_elbow_plot, py_fig

    # -------------------- Cluster Cells --------------------

    def cluster_cells(self):
        # Hide FutureWarnings.
        warnings.simplefilter('ignore',
                              FutureWarning) if self.verbose else None

        # -------------------- tSNE PLOT --------------------
        pc_sdev = pd.Series(np.std(self.data.obsm['X_pca'], axis=0))
        pc_sdev.index = pc_sdev.index + 1

        # Parameter values
        pc_range = range(2, 31)
        res_range = [
            float('{:0.1f}'.format(x)) for x in list(np.arange(.5, 2.1, 0.1))
        ]
        perp_range = range(5, min(51, len(self.data.obs_names)))
        # Default parameter values
        pcs = 10
        resolution = 1.2
        perplexity = 30

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
            height='800px',
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
                tsne_fig, py_tsne_fig = self._plot_tsne(figsize=(10, 9))

                display(
                    _create_export_button(
                        tsne_fig, '3_cluster_cells_tsne_plot'))
                py.iplot(py_tsne_fig, show_link=False)

                # close progress bar
                progress_bar.close()

        # Button widget
        go_button = Button(description='Plot', button_style='info')
        go_button.on_click(plot_tsne_callback)

        # Parameter descriptions
        param_info = _output_message('''
            <h3 style="position: relative; top: -10px">Clustering Parameters</h3>
            <p>
            <h4>Number of PCs (Principal Components)</h4>The number of principal components (PCs) to use in clustering.
            It is important to note that the fewer PCs we choose to use, the less noise we have when clustering,
            but at the risk of excluding relevant biological variance. Look at the plot in <b>Step 2</b> showing the
            percent varianced explained by each principle components and choose a cutoff where there is a clear elbow
            in the graph.<br><br>
            <h4>Resolution</h4>Higher resolution means more and smaller clusters. We find that values 0.6-1.2 typically
            returns good results for single cell datasets of around 3K cells. Optimal resolution often increases for
            larger datasets.<br><br>
            <h4>Perplexity</h4>The perplexity parameter loosely models the number of close neighbors each point has.
            <a href="https://distill.pub/2016/misread-tsne/">More info on how perplexity matters here</a>.
            </p>''')

        help_message = '''Hover over the plot to interact. Click and drag to zoom. Click on the legend to hide or show
        specific clusters; single-click hides/shows the cluster while double-click isolates the cluster.'''
        sliders = HBox([pc_slider, res_slider, perp_slider])
        ui = VBox(
            [param_info, _info_message(help_message), sliders, go_button])

        plot_box = VBox([ui, plot_output])
        display(plot_box)
        plot_tsne_callback()

        # Revert to default settings to show FutureWarnings.
        warnings.simplefilter('default',
                              FutureWarning) if self.verbose else None

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

        self.data.obs['louvain_groups']

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
        fig, ax = plt.subplots(1, 1, figsize=figsize)
        for c, group in tsne_coordinates.groupby(by='colors'):
            group.plot(
                x='tSNE_1',
                y='tSNE_2',
                kind='scatter',
                c=colors[c],
                label=c,
                alpha=0.7,
                ax=ax,
                legend=True)

        plt.title('tSNE Visualization', size=16)
        ax.set_xlabel(ax.get_xlabel(), size=12)
        ax.set_ylabel(ax.get_ylabel(), size=12)
        plt.tight_layout()
        plt.close()

        py_fig = tls.mpl_to_plotly(fig)
        # Let plotly generate legend for plotly fig
        py_fig['layout']['annotations'] = []
        py_fig['layout']['showlegend'] = True
        py_fig['layout']['legend'] = {'orientation': 'h'}

        py_fig.update(data=[dict(name=c) for c in cluster_names])

        return fig, py_fig

    # -------------------- MARKER ANALYSIS --------------------

    def visualize_top_markers(self):
        warnings.simplefilter('ignore',
                                FutureWarning) if self.verbose else None

        ## Commonly used data
        cell_clusters = self.data.obs['louvain_groups'].astype(int)
        cluster_names = np.unique(cell_clusters).tolist()
        cluster_names.sort(key=int)
        heatmap_text = HTML('''
        <h3>Visualize Top Markers</h3>
        <p style="font-size:14px; line-height:{};">Show the top markers for each cluster as a heatmap.</p>
        '''.format(_LINE_HEIGHT))
        heatmap_n_markers = IntSlider(
            description="# markers", value=5, min=5, max=100, step=5)
        heatmap_test = Dropdown(
            description='test',
            options=['wilcoxon', 't-test'],
            value='wilcoxon')
        heatmap_plot_button = Button(description='Plot', button_style='info')
        heatmap_header_box = VBox()
        heatmap_header_box.children = [
            heatmap_text, _info_message(
                'Double-click the heatmap to zoom in and scroll for more detail.'
            ), heatmap_n_markers, heatmap_test, heatmap_plot_button
        ]

        marker_heatmap_output = Output()

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
                        '4_visualize_top_markers_heatmap_plot'
                    ))

                display(fig)
                top_marker_progress_bar.close()

        heatmap_plot_button.on_click(plot_heatmap)

        heatmap_box = VBox([heatmap_header_box, marker_heatmap_output]) 

        display(heatmap_box)

        plot_heatmap()

    def visualize_marker_expression(self):
        marker_box = VBox() #box that will contain tSNEs and violins
        markers_header_box = VBox() #box for the header description and button
        ## code for making the header and button
        markers_plot_box = HBox()
        marker_box.children = [markers_header_box, markers_plot_box]

        gene_input_description = HTML('''<h3>Visualize Marker Expression</h3>
                                      <p style="font-size:14px; line-height:{};">Visualize the expression of gene(s) in each cell projected on the t-SNE map and the distribution across identified clusters.
                                         Provide any number of genes. If more than one gene is provided, the average expression of those genes will be shown.</p>
                                      '''.format(_LINE_HEIGHT))
        gene_input = Text("CD14")
        update_button = Button(
            description='Plot Expression', button_style='info')
        gene_input_box = HBox([gene_input, update_button])

        markers_header_box.children = [gene_input_description, gene_input_box]

        tsne_box = Output()
        violin_box = Output()
        markers_plot_box.children = [tsne_box, violin_box]

        def check_gene_input(t):
            '''Don't allow submission of empty input.'''
            if gene_input.value == '':
                update_button.disabled = True
            else:
                update_button.disabled = False

        def update_query_plots(b):
            # Format gene list. Split by comma, remove whitespace, then split by whitespace.
            tsne_box.clear_output()
            violin_box.clear_output()

            gene_list = str(gene_input.value).upper()
            gene_list = gene_list.split(',')
            gene_list = [s.split(' ') for s in gene_list]
            gene_list = np.concatenate(gene_list).ravel().tolist()
            gene_list = [gene for gene in gene_list if gene.strip()]

            if len(gene_list) == 1:
                gene_list = [gene_list[0]]

            # Retrieve expression
            gene_locs = []
            for gene in gene_list:
                if gene in self.data.raw.var_names:
                    gene_locs.append(self.data.raw.var_names.get_loc(gene))
                else:
                    # Gene not found
                    tsne_box.clear_output()
                    with tsne_box:
                        display(
                            _warning_message(
                                'The gene <code>{}</code> was not found. Try again.'.
                                format(gene)))
                    return
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
            with tsne_box:

                display(tab1_progress_bar)

                
                # generate tSNE markers plot
                tsne_markers_fig = self._plot_tsne_markers(title, values, (6, 6))
                tsne_markers_py_fig = tls.mpl_to_plotly(tsne_markers_fig)

                # Hide progress bar
                tab1_progress_bar.close()

                display(
                    _create_export_button(
                        tsne_markers_fig,
                        '4_visualize_marker_tsne_plot'))
                py.iplot(tsne_markers_py_fig, show_link=False)
                
            # Violin plots
            
            with violin_box:
                marker_violin_plot = self._plot_violin_plots(title, values)
                display(
                    _create_export_button(
                        marker_violin_plot,
                        '4_visualize_marker_violin_plot'))
                display(
                    HTML('<h3>{} Expression Across Clusters</h3>'.format(
                        title)))
                display(marker_violin_plot)

        gene_input.observe(check_gene_input)
        gene_input.on_submit(update_query_plots)
        update_button.on_click(update_query_plots)

        display(marker_box)

        update_query_plots("CD14")       

    def visualize_markers(self):
        # Hide FutureWarnings
        warnings.simplefilter('ignore',
                                FutureWarning) if self.verbose else None

        ## Commonly used data
        cell_clusters = self.data.obs['louvain_groups'].astype(int)
        cluster_names = np.unique(cell_clusters).tolist()
        cluster_names.sort(key=int)

        marker_box = VBox() #box that will contain tSNEs and violins
        markers_header_box = VBox() #box for the header description and button
        ## code for making the header and button
        markers_plot_box = HBox()
        marker_box.children = [markers_header_box, markers_plot_box]

        gene_input_description = HTML('''<h3>Visualize Markers</h3>
                                      <p style="font-size:14px; line-height:{};">Visualize the expression of gene(s) in each cell projected on the t-SNE map and the distribution across identified clusters.
                                         Provide any number of genes. If more than one gene is provided, the average expression of those genes will be shown.</p>
                                      '''.format(_LINE_HEIGHT))
        gene_input = Text("CD14")
        update_button = Button(
            description='Plot Expression', button_style='info')
        gene_input_box = HBox([gene_input, update_button])

        markers_header_box.children = [gene_input_description, gene_input_box]

        tsne_box = Output()
        violin_box = Output()
        markers_plot_box.children = [tsne_box, violin_box]

        def check_gene_input(t):
            '''Don't allow submission of empty input.'''
            if gene_input.value == '':
                update_button.disabled = True
            else:
                update_button.disabled = False

        def update_query_plots(b):
            # Format gene list. Split by comma, remove whitespace, then split by whitespace.
            tsne_box.clear_output()
            violin_box.clear_output()

            gene_list = str(gene_input.value).upper()
            gene_list = gene_list.split(',')
            gene_list = [s.split(' ') for s in gene_list]
            gene_list = np.concatenate(gene_list).ravel().tolist()
            gene_list = [gene for gene in gene_list if gene.strip()]

            if len(gene_list) == 1:
                gene_list = [gene_list[0]]

            # Retrieve expression
            gene_locs = []
            for gene in gene_list:
                if gene in self.data.raw.var_names:
                    gene_locs.append(self.data.raw.var_names.get_loc(gene))
                else:
                    # Gene not found
                    tsne_box.clear_output()
                    with tsne_box:
                        display(
                            _warning_message(
                                'The gene <code>{}</code> was not found. Try again.'.
                                format(gene)))
                    return
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
            with tsne_box:

                display(tab1_progress_bar)

                
                # generate tSNE markers plot
                tsne_markers_fig = self._plot_tsne_markers(title, values, (6, 6))
                tsne_markers_py_fig = tls.mpl_to_plotly(tsne_markers_fig)

                # Hide progress bar
                tab1_progress_bar.close()

                display(
                    _create_export_button(
                        tsne_markers_fig,
                        '4_visualize_marker_tsne_plot'))
                py.iplot(tsne_markers_py_fig, show_link=False)
                
            # Violin plots
            
            with violin_box:
                marker_violin_plot = self._plot_violin_plots(title, values)
                display(
                    _create_export_button(
                        marker_violin_plot,
                        '4_visualize_marker_violin_plot'))
                display(
                    HTML('<h3>{} Expression Across Clusters</h3>'.format(
                        title)))
                display(marker_violin_plot)

        gene_input.observe(check_gene_input)
        gene_input.on_submit(update_query_plots)
        update_button.on_click(update_query_plots)

        display(marker_box)

        update_query_plots("CD14")


        heatmap_text = HTML('''
        <h3>Visualize Top Markers</h3>
        <p style="font-size:14px; line-height:{};">Show the top markers for each cluster as a heatmap.</p>
        '''.format(_LINE_HEIGHT))
        heatmap_n_markers = IntSlider(
            description="# markers", value=10, min=5, max=100, step=5)
        heatmap_test = Dropdown(
            description='test',
            options=['wilcoxon', 't-test'],
            value='wilcoxon')
        heatmap_plot_button = Button(description='Plot', button_style='info')
        heatmap_header_box = VBox()
        heatmap_header_box.children = [
            heatmap_text, _info_message(
                'Double-click the heatmap to zoom in and scroll for more detail.'
            ), heatmap_n_markers, heatmap_test, heatmap_plot_button
        ]

        marker_heatmap_output = Output()

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
                        '4_visualize_top_markers_heatmap_plot'
                    ))

                display(fig)
                top_marker_progress_bar.close()

        heatmap_plot_button.on_click(plot_heatmap)

        heatmap_box = VBox([heatmap_header_box, marker_heatmap_output]) 

        display(heatmap_box)

        plot_heatmap()

    def OLD_visualize_markers(self):
        # Hide FutureWarnings.
        warnings.simplefilter('ignore',
                              FutureWarning) if self.verbose else None

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
            align_items='flex-start',
            height='800px',
            width='100%',
            padding='0',
            overflow_y='auto'))
        marker_heatmap_output = Output(
            style='border: 1px solid green',
            layout=Layout(
                display='flex',
                justify_content='flex-start',
                align_items='flex-start',
                height='1000px',
                width='100%',
                margin='0',
                overflow_x='auto',
                overflow_y='auto'))

        # Create main container
        main_box = Tab(layout=Layout(
            padding='0 12px', flex='1', min_width='500px'))

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
        main_box.children = [marker_plot_box, heatmap_box]

        # TODO name tabs
        main_box.set_title(0, 'tSNE Plot')
        main_box.set_title(1, 'Heatmap')
        # Table
        explore_markers_box = Accordion(layout=Layout(
            max_width='425px', orientation='vertical'))
        cluster_table_header_box = VBox()
        explore_markers_box.children = [
            VBox([cluster_table_header_box, marker_table_output])
        ]
        explore_markers_box.set_title(0, 'Explore Markers')

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
        gene_input = Text("CD14")
        update_button = Button(
            description='Plot Expression', button_style='info')
        gene_input_box = HBox([gene_input, update_button])
        main_header_box.children = [gene_input_description, gene_input_box]

        def check_gene_input(t):
            '''Don't allow submission of empty input.'''
            if gene_input.value == '':
                update_button.disabled = True
            else:
                update_button.disabled = False

        def update_query_plots(b):
            # Format gene list. Split by comma, remove whitespace, then split by whitespace.
            gene_list = str(gene_input.value).upper()
            gene_list = gene_list.split(',')
            gene_list = [s.split(' ') for s in gene_list]
            gene_list = np.concatenate(gene_list).ravel().tolist()
            gene_list = [gene for gene in gene_list if gene.strip()]

            if len(gene_list) == 1:
                gene_list = [gene_list[0]]

            # Retrieve expression
            gene_locs = []
            for gene in gene_list:
                if gene in self.data.raw.var_names:
                    gene_locs.append(self.data.raw.var_names.get_loc(gene))
                else:
                    # Gene not found
                    marker_plot_tab_1_output.clear_output()
                    with marker_plot_tab_1_output:
                        display(
                            _warning_message(
                                'The gene <code>{}</code> was not found. Try again.'.
                                format(gene)))
                    return
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
                tsne_markers_fig = self._plot_tsne_markers(title, values, (6, 6))
                tsne_markers_py_fig = tls.mpl_to_plotly(tsne_markers_fig)

                # Hide progress bar
                tab1_progress_bar.close()

                display(
                    _create_export_button(
                        tsne_markers_fig,
                        '4_visualize_marker_tsne_plot'))
                py.iplot(tsne_markers_py_fig, show_link=False)
                

            marker_plot_tab_2_output.clear_output()
            tab2_progress_bar = _create_progress_bar()
            with marker_plot_tab_2_output:
                display(tab2_progress_bar)

                # generate tSNE clusters plot
                tsne_fig, tsne_py_fig = self._plot_tsne(figsize=(6.5, 7))

                display(
                    _create_export_button(
                        tsne_fig, '4_visualize_analysis_tsne_plot'))
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
                        '4_visualize_marker_violin_plot'))
                display(
                    HTML('<h3>{} Expression Across Clusters</h3>'.format(
                        title)))
                display(marker_violin_plot)

        gene_input.observe(check_gene_input)
        gene_input.on_submit(update_query_plots)
        update_button.on_click(update_query_plots)

        # ------------------------- Heatmap -------------------------

        heatmap_text = HTML('''
        <h3>Visualize Top Markers</h3>
        <p style="font-size:14px; line-height:{};">Show the top markers for each cluster as a heatmap.</p>
        '''.format(_LINE_HEIGHT))
        heatmap_n_markers = IntSlider(
            description="# markers", value=10, min=5, max=100, step=5)
        heatmap_test = Dropdown(
            description='test',
            options=['wilcoxon', 't-test'],
            value='wilcoxon')
        heatmap_plot_button = Button(description='Plot', button_style='info')
        heatmap_header_box.children = [
            heatmap_text, _info_message(
                'Double-click the heatmap to zoom in and scroll for more detail.'
            ), heatmap_n_markers, heatmap_test, heatmap_plot_button
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
                        '4_visualize_top_markers_heatmap_plot'
                    ))

                display(fig)
                top_marker_progress_bar.close()

        heatmap_plot_button.on_click(plot_heatmap)

        # ------------------------- Cluster Table -------------------------
        cluster_table_header = HTML('''
        <p style="font-size:14px; line-height:{};">Test for differentially expressed genes between subpopulations of cells.</p>'''
                                    .format(_LINE_HEIGHT))

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
        cluster_table_button = Button(
            description='Explore', button_style='info')
        cluster_table_note = HTML('''
            <hr>
            <h4>Output Table Info</h4>
            <p style="font-size:14px; line-height:{};">
            <ul style="list-style-position: inside; padding-left: 0; font-size:14px; line-height:{};">
            <li><code>Gene</code>: the gene name<br></li>
            <li><code>adj.pval</code>: Benjamini & Hochberg procedure adjusted p-values<br></li>
            <li><code>logFC</code>: log fold-change of average relative expression of gene in the first group compared to the second group<br></li>
            <li><code>%.expr.c#</code>: # of cells in the first group that express the gene<br></li>
            <li><code>%.expr.c#</code>: # of cells in the second group that express the gene<br></li>
            </ul>
            <hr>
            '''.format(_LINE_HEIGHT, _LINE_HEIGHT))

        cluster_param_header = HTML('<h4>Compare Clusters</h4>')

        def update_cluster_table(b=None):
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
            _info_message('Hide/show this panel by clicking the <b>Explore Markers</b> header above.'),
            cluster_table_header, cluster_table_note, _info_message(
                'Export the table using the menu, which can be accessed in the top left hand corner of the "Gene" column.'
            ), cluster_param_header, cluster_param_box, param_test, cluster_table_button
        ]

        # ------------------------- Main Table -------------------------

        # Configure layout
        top_box = HBox([main_box, explore_markers_box])
        display(top_box)

        update_query_plots("CD14")
        plot_heatmap()
        update_cluster_table()


        # Revert to default settings to show FutureWarnings.
        warnings.simplefilter('default',
                              FutureWarning) if self.verbose else None

    def _plot_tsne_markers(self, title, gene_values, figsize):
        fig, ax = plt.subplots(1, 1, figsize=figsize)
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
        fig = plt.figure(figsize=(8, 5))
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
        if ident_2 == 'rest':
            pct_expr_2_prefix = '%.expr.'
        else:
            pct_expr_2_prefix = '%.expr.c'
        results = pd.DataFrame(
            [marker_names, marker_scores, log_fc, pct_1, pct_2],
            index=[
                'Gene', 'adj.pval', 'logFC', '%.expr.c{}'.format(ident_1),
                '{}{}'.format(pct_expr_2_prefix, ident_2)
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
                    maxColor='white')
            else:
                highlighter = TableDisplayCellHighlighter.getHeatmapHighlighter(
                    c,
                    TableDisplayCellHighlighter.SINGLE_COLUMN,
                    minColor='white',
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
        cmap_binary = {0: 'grey', 1: 'lightgrey'}
        cell_colors = pd.Series(group_labels, index=counts.columns).map(cmap)
        gene_labels = np.array([[c] * num_markers
                                for c in cluster_names]).flatten()
        gene_labels = [label % 2 for label in gene_labels]
        gene_colors = pd.Series(gene_labels).map(cmap_binary).tolist()

        dim = max(12, 12*num_markers/10)
        # Heatmap
        g = sns.clustermap(
            counts,
            row_cluster=False,
            col_cluster=False,
            row_colors=gene_colors,
            col_colors=cell_colors,
            xticklabels=False,
            figsize=(dim, dim),
            cmap=_EXPRESSION_CMAP)

        # Cell cluster legend
        for label in cluster_names:
            g.ax_col_dendrogram.bar(
                0, 0, color=cmap[label], label=label, linewidth=0)
        g.ax_col_dendrogram.legend(loc="center", ncol=8)

        # Reposition colorbar
        g.cax.set_position([.15, .2, .03, .45])

        hm = g.ax_heatmap
        hm.set_yticklabels(counts.index, {'fontsize': '9'})
        hm.set_yticks([x + 0.5 for x in range(len(counts.index))])

        hm.set_xlabel("Cells")
        hm.xaxis.set_label_coords(0.5, 1.1)

        hm.set_ylabel("Top {} Genes of each Cluster".format(num_markers))
        hm.yaxis.set_label_coords(-0.1, 0.5)

        plt.close()
        return g.fig

    # -------------------- FILE EXPORT --------------------

    def export_data(self, path, h5ad=False):
        # Hide "omitting to write sparse annotation message" from scanpy.
        stat = _get_new_status("Preparing to export your data...")
        display(stat)
        warnings.simplefilter('ignore', UserWarning)
        if h5ad:
            # Assume same directory if simple filename
            if '/' not in path:
                path = './' + path

            # Append .h5ad suffix
            if not path.endswith('.h5ad'):
                path = path + '.h5ad'
            _update_status(stat, "Writing data to disk...")
            self.data.write(path)

            path_message = '<code>{}</code> in .h5ad format.'.format(path)
        else:
            # Export AnnData object as series of csv files.
            _update_status(stat, "Writing data to disk...")
            self.data.write_csvs(path, skip_data=False)
            path_message = 'the <a href="{}" target="_blank">{}</a> folder as <code>.csv</code> files.'.format(
                path, path)

        _update_status(stat, "Done! Exported data to {}".format(path_message))

        # User feedback
        #display(
        #    _output_message('''
        #    <h3 style="position: relative; top: -10px">Results</h3>
        #    <p style="font-size:14px; line-height:20px;">Exported data to {}</p>
        #    </div>'''.format(path_message)))

        # Turn user warnings back on.
        warnings.simplefilter('default', UserWarning)
