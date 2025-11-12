import os
import win32com.client as win32
import traceback
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re
import networkx as nx
from typing import List, Tuple, Optional
# import pyvis.network as Network
import webbrowser


def Aspen_loading():
    try:
        app = win32.Dispatch('Apwn.Document')
    except Exception:
        app = win32.gencache.EnsureDispatch('Apwn.Document')
    return app

class AspenPlus_Connector:
    def __init__(self):
        super().__init__()
        self.app = None
    
    def get_start(self, Aspen_file_path):
        if self.app is None:
            self.app = Aspen_loading()
            self.app.InitFromArchive2(os.path.abspath(Aspen_file_path))
            self.app.Visible = True
    
    def find_node(self, address):
        try:
            node = self.app.Tree.FindNode(address)
            return node
        except Exception:
            return None

    def get_stream_info(self, stream_id):
        out_info = {}

        address_stream = [
            rf"\Data\Streams\{stream_id}\Output\PRES_OUT\MIXED", # pressure
            rf"\Data\Streams\{stream_id}\Output\TEMP_OUT\MIXED", # temperature
            rf"\Data\Streams\{stream_id}\Output\MASSFLMX\MIXED", # total mass flowrate
        ]
        for idx in address_stream:
            node = self.app.Tree.FindNode(idx)
            if node is not None:
                try:
                    val = node.Value
                except Exception:
                    try:
                        val = float(node)
                    except Exception:
                        val = None
                if 'temp' in idx.lower():
                    out_info['Temperature/°C'] = val
                elif 'pres' in idx.lower():
                    out_info['Pressure/Bar'] = val
                elif 'mass' in idx.lower():
                    out_info['Massflow/(kg/h)'] = val
        
        return out_info
    
    def list_all_streams(self, pattern=r"^[A-Za-z0-9\-]+$"):
        root_stream = self.app.Tree.FindNode(r"\Data\Streams")
        streams = []
        if root_stream is None:
            print(root_stream)
            return streams
        try:
            elements = root_stream.Elements
            for index in range(elements.Count):
                node = elements.Item(index)
                name = getattr(node, 'Name', None)
                if name and re.match(pattern, name):
                    streams.append(name)
        except Exception:
            pass
        return streams
    
    def list_all_blocks(self, pattern=r"^[A-Za-z0-9\-]+$"):
        root_block = self.app.Tree.FindNode(r"\Data\Blocks")
        blocks = []
        if root_block is None:
            print(root_block)
            return blocks
        try:
            elements = root_block.Elements
            for index in range(elements.Count):
                node = elements.Item(index)
                name = getattr(node, 'Name', None)
                if name and re.match(pattern, name):
                    blocks.append(name)
        except Exception:
            pass
        return blocks
    
    def list_all_components(self, pattern=r"^[A-Za-z0-9_\-\(\)\,\.\+]+$"):
        root_component = self.app.Tree.FindNode("\Data\Components\Specifications\Input\ANAME")
        components = []
        if root_component is None:
            print(root_component)
            return components
        try:
            elements = root_component.Elements
            for index in range(elements.Count):
                node = elements.Item(index)
                name = getattr(node, "Name", None)
                if name and re.match(pattern, name):
                    components.append(name)
        except Exception as e:
            print(e)
        return components
    
    def get_all_massfrac(self, stream, component_set):
        out_massfrac_info = {}
        for component_id in component_set:
            address = rf"\Data\Streams\{stream}\Output\MASSFRAC\MIXED\{component_id}"
            node = self.app.Tree.FindNode(address)
            if node is not None:
                try:
                    val = node.Value
                except Exception:
                    try:
                        val = float(node)
                    except Exception:
                        val = None
                out_massfrac_info[component_id] = val
        return out_massfrac_info

    def get_connection_info(self, stream_ids):
        connection_set = []
        for s in stream_ids:
            root = self.app.Tree.FindNode(f"\Data\Streams\{s}\Connections")
            if root is not None:
                for i in range(root.Elements.Count):
                    if root.Elements.Item(i).Value == 'DEST':
                        Stream_target = root.Elements.Item(i).Name
                    elif root.Elements.Item(i).Value == 'SOURCE':
                        Stream_source = root.Elements.Item(i).Name
                    else:
                        pass
                connection_set.append((s, Stream_source, Stream_target))
        return connection_set
        
    def Quit_Aspen(self):
        self.app.Quit()
        # self.app.Close()

def stream_to_matrix(stream_set):
    df = pd.DataFrame.from_dict(stream_set, orient='index')
    df = df.fillna(0)
    return df

def del_all_zeros(df): # delete all columns with all zeros
    tolerance = 1e-10
    matrix_nozero = df.loc[:, (df !=0).any(axis=0)]
    # matrix_nozero = df.loc[:, (df.abs() > tolerance).any(axis=0)]
    return matrix_nozero

def get_stream_matrix(self, stream_mass_matrix, stream_property_matrix):
    merged = pd.concat([stream_mass_matrix, stream_property_matrix], axis=1)
    return merged

def set_virtual_node(connection_set): # connection_set = [('edge', 'edge source node', 'edge target node')]
    source_count = 0
    target_count = 0
    updated_connections = [] # Save updated connections

    for edge, start, end in connection_set:
        if start == "#1":
            source_count += 1
            new_node = f'Virtual_source_node_{source_count}'
            start = new_node
        
        if end == "#0":
            target_count += 1
            new_node = f'Virtual_target_node_{target_count}'
            end = new_node
        
        updated_connections.append((edge, start, end))
    
    return updated_connections


# ========================================================================================
# def build_aspen_graph(connection_set):
#     G = nx.DiGraph()
#     for edge, source, end in connection_set:
#         G.add_node(source)
#         G.add_node(end)
#         G.add_edge(source, end, label=edge)
#     return G

# def plot_aspen_Graph(
#         G: nx.DiGraph,
#         layout: str = 'spring',
#         figsize: Tuple = (7, 7),
#         title: Optional[str] = None,
#         show_edge_labels: bool = True,
#         seed: int = 42,
#         save_path: Optional[str] = None,
# ) -> None:
    
#     if G.number_of_edges() == 0:
#         print("No edges found in the graph")
#         return
#     if layout == "spring":
#         pos = nx.spring_layout(G, seed=seed)
#     elif layout == "shell":
#         pos = nx.shell_layout(G)
#     elif layout == "circular":
#         pos = nx.circular_layout(G)
#     elif layout == "kamada_kawai":
#         pos = nx.kamada_kawai_layout(G)
#     else:
#         pos = nx.spring_layout(G, seed=seed)

#     plt.figure(figsize=figsize)
#     if title:
#         plt.title(title)

#     nx.draw_networkx_nodes(G, pos, node_size=300)
#     nx.draw_networkx_edges(G, pos, arrows=True, arrowstyle='-|>', arrowsize=16)
#     nx.draw_networkx_labels(G, pos, font_size=10, font_weight='bold')

#     if show_edge_labels:
#         edge_labels = nx.get_edge_attributes(G, 'label')
#         nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, label_pos=0.5, font_size=9)

#     plt.axis('off')
#     plt.tight_layout()
#     if save_path:
#         plt.savefig(save_path, bbox_inches='tight')
#         print(f"Saved plot to: {save_path}")
#     plt.show()


# ======================================================================
# if __name__ == '__main__':
#     aspen = AspenPlus_Connector()
#     aspen.get_start('NITRO.bkp')
#     stream_ids = aspen.list_all_streams()
#     block_ids = aspen.list_all_blocks()
#     print("Found streams:", stream_ids)
#     print("Found blocks:", block_ids)

#     stream_info = {}
#     for s in stream_ids:
#         info = aspen.get_stream_info(s)
#         if info:
#             stream_info[s] = info
#             print(f"Stream {s}: T={info.get('Temperature/°C')}°C, P={info.get('Pressure/Bar')}Bar, flowrate={info.get('Massflow/(kg/h)')}kg/h")
    
    # aspen.Quit_Aspen()
