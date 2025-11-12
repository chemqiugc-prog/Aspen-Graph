import numpy as np
import pandas as pd
import os
import math
import json
from typing import List, Tuple, Optional, Callable, Dict, Any
from scipy.stats import entropy, wasserstein_distance

eps = 1e-10

class AspenNodeImportance_Calculator:

    @staticmethod
    def js_divergence(p, q): # Jensen-Shannon divergence
        p = np.array(p, dtype=float) + eps
        q = np.array(q, dtype=float) + eps
        p /= np.sum(p)
        q /= np.sum(q)
        m = 0.5 * (p + q)
        return 0.5 * (entropy(p, m) + entropy(q, m))

    @staticmethod
    def weighted_hist(values, weights, bins):
        # Compute weighted histogram with normalization
        hist, _ = np.histogram(values, bins=bins, weights=weights, density=True)
        hist = hist.astype(float)
        if hist.sum() == 0:
            return np.ones(len(hist)) / len(hist)
        return hist / np.sum(hist)

    @staticmethod
    def kl_divergence(mu1, s1, mu2, s2):
        # KL divergence between two normal distributions
        s1 = max(s1, eps)
        s2 = max(s2, eps) # avoid zero division

        def kl(m1, std1, m2, std2):
            return math.log(std2 / std1) + (std1 ** 2 + (m1 - m2) ** 2) / (2 * std2 ** 2) - 0.5
        
        return 0.5 * (kl(mu1, s1, mu2, s2) + kl(mu2, s2, mu1, s1))
    
    @staticmethod
    def clr_transform(comp):
        # Centered log-ratio transform
        v = np.array(comp, dtype=float) + eps
        v = v / v.sum()
        gm = np.exp(np.mean(np.log(v)))
        return np.log(v / gm)

    @staticmethod
    def aitchison_distance(c1, c2): # Aitchison distance between two compositions
        return np.linalg.norm(AspenNodeImportance_Calculator.clr_transform(c1) - 
                              AspenNodeImportance_Calculator.clr_transform(c2))
    
    @staticmethod
    def weighted_sample(vals, wts, cap=500):
        vals = np.array(vals)
        if len(vals) == 0: 
            return np.array([])
        wts = np.array(wts, dtype=float)
        if np.isnan(wts.sum()) or wts.sum() <= 0:
            return np.array([])
        norm = wts / wts.sum()

        total = min(cap, max(1, int(wts.sum() / max(1.0, np.mean(wts)))))
        counts = np.maximum(1, np.round(norm * total)).astype(int)
        counts = counts[:len(vals)]
        samples = np.concatenate([np.repeat(v, int(c)) for v, c in zip(vals, counts) if int(c) > 0])
        return samples

    @staticmethod
    def parse_composition_string(val):
        # Parse string or list representing composition into a normalized numeric vector
        if val is None or (isinstance(val, float) and np.isnan(val)):
            return None
        if isinstance(val, (list, tuple, np.ndarray)):
            arr = np.array(val, dtype=float)
            s = arr.sum()
            if s == 0: 
                return (arr + eps) / (arr.size)
            return (arr / s).tolist() 
        
        s = str(val).strip()  
        
        try:
            parsed = json.loads(s)
            if isinstance(parsed, list):
                arr = np.array(parsed, dtype=float)
                ssum = arr.sum()
                if ssum == 0:
                    return (arr + eps) / len(arr)
                return (arr / ssum).tolist()
        except Exception:
            pass
        
        for sep in ['|', ';', ',', ' ']:
            if sep in s:
                parts = [p.strip() for p in s.split(sep) if p.strip() != '']
                try:
                    nums = np.array([float(p) for p in parts], dtype=float)
                    ssum = nums.sum()
                    if ssum == 0:
                        return (nums + eps) / len(nums)
                    return (nums / ssum).tolist()
                except Exception:
                    continue  
        
        try:
            v = float(s)
            return [v]
        except Exception:
            return None  

    # ==============================================================================
    # Default virtual node detector
    # ==============================================================================

    @staticmethod
    def default_is_virtual(node_name: str) -> bool:
        if node_name is None:
            return False
        s = str(node_name).strip()
        if s == '':
            return True
        if s.startswith('#'):
            return True
        if 'virtual' in s.lower():
            return True
        return False

    def transform_edges_from_dataframe(
        self,
        edges_df: pd.DataFrame,
        connections: List[Tuple[str, str, str]],
        edge_name_col: Optional[str] = None,
        pressure_cols_hint: Optional[List[str]] = None,
        temperature_cols_hint: Optional[List[str]] = None,
        flow_cols_hint: Optional[List[str]] = None,
        composition_cols_hint: Optional[List[str]] = None
    ) -> pd.DataFrame:

        conn_df = pd.DataFrame(connections, columns=['edge','from','to'])
        conn_df['edge'] = conn_df['edge'].astype(str).str.strip()
        edges = edges_df.copy()

        if edge_name_col and edge_name_col in edges.columns:
            edges[edge_name_col] = edges[edge_name_col].astype(str).str.strip()
            key_col = edge_name_col
        else:
            key_col = None
            for c in edges.columns:
                try:
                    if edges[c].astype(str).isin(conn_df['edge']).any():
                        key_col = c
                        break
                except Exception:
                    continue
            if key_col is None:
                if len(edges) == len(conn_df):
                    edges = edges.reset_index(drop=True)
                    edges['edge_key_for_merge'] = conn_df['edge'].astype(str).values
                    key_col = 'edge_key_for_merge'
                else:
                    raise ValueError("Cannot automatically determine edge key column. Please provide edge_name_col or ensure edges_df and connections are aligned by row order.")

        edges[key_col] = edges[key_col].astype(str).str.strip()

        cols_lower = {c.lower(): c for c in edges.columns}
        
        def pick_col(hints, fallback_keywords):
            """
            Select a column based on hints and fallback keywords.
            """
            if hints: 
                for h in hints:
                    if h in edges.columns:
                        return h
            for kw in fallback_keywords:
                for c_lower, c in cols_lower.items():
                    if kw in c_lower: 
                        return c
            return None 

        pressure_col = pick_col(pressure_cols_hint, ['pressure','pres','p_'])
        temperature_col = pick_col(temperature_cols_hint, ['temp','temperature','t_'])
        flow_col = pick_col(flow_cols_hint, ['flow','massflow','mass_flow','mflow'])

        comp_cols = composition_cols_hint if composition_cols_hint else None
        if comp_cols:
            comp_cols = [c for c in comp_cols if c in edges.columns]
            if len(comp_cols) == 0:
                comp_cols = None

        comp_list_col = None 
        if comp_cols is None:
            numeric_cols = edges.select_dtypes(include=[np.number]).columns.tolist()
            exclude = set(filter(None, [pressure_col, temperature_col, flow_col, key_col]))
            cand = [c for c in numeric_cols if c not in exclude] 
            species_cols = [c for c in cand if ((edges[c] >= -1e-6) & (edges[c] <= 1+1e-6)).mean() > 0.4]
            if len(species_cols) >= 2:
                comp_cols = species_cols
            else:
                list_like = [c for c in edges.columns if 
                           edges[c].astype(str).str.contains('\\[').any() or 
                           edges[c].astype(str).str.contains(',').any()]
                comp_list_col = list_like[0] if list_like else None

        merged = pd.merge(conn_df, edges, left_on='edge', right_on=key_col, how='left', suffixes=('','_m'))
        final_edges = merged[['edge','from','to']].copy() 

        if flow_col and flow_col in merged.columns:
            final_edges['flow_rate'] = pd.to_numeric(merged[flow_col], errors='coerce').fillna(0.0)
        else:
            final_edges['flow_rate'] = 0.0 

        if temperature_col and temperature_col in merged.columns:
            final_edges['temperature'] = pd.to_numeric(merged[temperature_col], errors='coerce')
        else:
            final_edges['temperature'] = np.nan 

        if pressure_col and pressure_col in merged.columns:
            final_edges['pressure'] = pd.to_numeric(merged[pressure_col], errors='coerce')
        else:
            final_edges['pressure'] = np.nan 

        if comp_cols:
            comps = merged[comp_cols].fillna(0.0).values.astype(float)
            sums = comps.sum(axis=1)  
            sums[sums == 0] = 1.0  
            comps_norm = (comps.T / sums).T.tolist() 
            final_edges['composition'] = comps_norm
        else:
            if comp_list_col and comp_list_col in merged.columns:
                final_edges['composition'] = merged[comp_list_col].apply(
                    AspenNodeImportance_Calculator.parse_composition_string)
            else:
                final_edges['composition'] = None 

        final_edges['flow_rate'] = final_edges['flow_rate'].astype(float)
        final_edges['temperature'] = pd.to_numeric(final_edges['temperature'], errors='coerce')
        final_edges['pressure'] = pd.to_numeric(final_edges['pressure'], errors='coerce')

        return final_edges

    # =============================
    # Node importance computation core
    # =============================

    def compute_node_importance(
        self,
        final_edges: pd.DataFrame,
        bins: int = 20,
        skip_virtual: bool = True,
        is_virtual_fn: Optional[Callable[[str], bool]] = None,
        feature_weights: Optional[Dict[str, Dict[str, float]]] = None
    ) -> pd.DataFrame:
        """
        Compute node importance metrics based on the differences between incoming and outgoing edges
        in terms of flowrate, temperature, pressure, and composition.
        """
        if is_virtual_fn is None:
            is_virtual_fn = AspenNodeImportance_Calculator.default_is_virtual

        # Default feature weights
        if feature_weights is None:
            feature_weights = {
                'flow_rate': {'js':1.0, 'w1':0.8, 'mean_diff_norm':0.5, 'gauss_kl_sym':0.6, 'entropy_diff':0.3},
                'temperature': {'js':1.2, 'w1':0.9, 'mean_diff_norm':0.6, 'gauss_kl_sym':0.7, 'entropy_diff':0.3},
                'pressure': {'js':0.9, 'w1':0.7, 'mean_diff_norm':0.4, 'gauss_kl_sym':0.5, 'entropy_diff':0.2},
                'composition': {'comp_aitchison':1.3, 'comp_js':1.0}
            }

        scalar_features = ['flow_rate','temperature','pressure']
        # Collect all unique nodes from 'from' and 'to' columns
        nodes = pd.unique(final_edges[['from','to']].values.ravel('K')).tolist()

        rows = []  # store results for each node
        
        # Normalize feature ranges
        ranges = {}
        for f in scalar_features:
            col = final_edges[f]
            valid = col.dropna()
            frange = float(valid.max() - valid.min()) if not valid.empty and valid.max() > valid.min() else 1.0
            ranges[f] = frange

        for nid in nodes:
            if skip_virtual and is_virtual_fn(nid):
                continue

            in_edges = final_edges[final_edges['to'] == nid]
            out_edges = final_edges[final_edges['from'] == nid]
            row = {'node_id': nid}

            # Basic degree and flow metrics
            row['in_degree'] = int(len(in_edges))
            row['out_degree'] = int(len(out_edges))
            row['total_in_flow'] = float(in_edges['flow_rate'].sum())
            row['total_out_flow'] = float(out_edges['flow_rate'].sum())

            # Compute scalar feature differences
            for f in scalar_features:
                # Weighted mean and std (using flowrate as weight)
                if len(in_edges) > 0 and in_edges['flow_rate'].sum() > 0:
                    mu_in = float(np.average(in_edges[f].fillna(0.0), 
                                           weights=in_edges['flow_rate'].where(in_edges['flow_rate']>0,1.0)))
                    std_in = float(in_edges[f].std()) if len(in_edges) > 1 else 0.0
                else:
                    mu_in, std_in = np.nan, np.nan
                    
                if len(out_edges) > 0 and out_edges['flow_rate'].sum() > 0:
                    mu_out = float(np.average(out_edges[f].fillna(0.0), 
                                            weights=out_edges['flow_rate'].where(out_edges['flow_rate']>0,1.0)))
                    std_out = float(out_edges[f].std()) if len(out_edges) > 1 else 0.0
                else:
                    mu_out, std_out = np.nan, np.nan

                row[f + '_mean_in'] = mu_in
                row[f + '_mean_out'] = mu_out
                frange = ranges[f]
                # Normalized mean difference
                row[f + '_mean_diff_norm'] = (0.0 if (math.isnan(mu_in) and math.isnan(mu_out)) 
                                             else abs((np.nan_to_num(mu_out) - np.nan_to_num(mu_in))) / frange)
                # Symmetric Gaussian KL divergence
                row[f + '_gauss_kl_sym'] = (0.0 if (math.isnan(mu_in) or math.isnan(mu_out)) 
                                           else AspenNodeImportance_Calculator.kl_divergence(
                                               mu_in, max(std_in,1e-9), mu_out, max(std_out,1e-9)))

                # Weighted histograms
                if len(in_edges) > 0:
                    p = AspenNodeImportance_Calculator.weighted_hist(
                        in_edges[f].fillna(0.0).values, in_edges['flow_rate'].values, 
                        bins=np.linspace(final_edges[f].min(), final_edges[f].max(), bins+1))
                else:
                    p = np.ones(bins)/bins
                    
                if len(out_edges) > 0:
                    q = AspenNodeImportance_Calculator.weighted_hist(
                        out_edges[f].fillna(0.0).values, out_edges['flow_rate'].values, 
                        bins=np.linspace(final_edges[f].min(), final_edges[f].max(), bins+1))
                else:
                    q = np.ones(bins)/bins
                    
                # JS divergence
                row[f + '_js'] = float(AspenNodeImportance_Calculator.js_divergence(p, q))
                
                # Wasserstein distance via weighted sampling
                s_in = (AspenNodeImportance_Calculator.weighted_sample(in_edges[f].fillna(0.0).values, 
                                                             in_edges['flow_rate'].values) 
                        if len(in_edges) > 0 else np.array([]))
                s_out = (AspenNodeImportance_Calculator.weighted_sample(out_edges[f].fillna(0.0).values, 
                                                              out_edges['flow_rate'].values) 
                         if len(out_edges) > 0 else np.array([]))
                row[f + '_w1'] = (0.0 if (s_in.size == 0 or s_out.size == 0) 
                                 else float(wasserstein_distance(s_in, s_out)))
                
                # Entropy calculations
                row[f + '_entropy_in'] = float(entropy(p + eps))
                row[f + '_entropy_out'] = float(entropy(q + eps))
                row[f + '_entropy_diff'] = row[f + '_entropy_out'] - row[f + '_entropy_in']

            # Composition-based metrics
            def weighted_comp_mean(df):
                """Compute flow-weighted mean of compositions"""
                if len(df) == 0:
                    return None
                comps = [c for c in df['composition'].values if isinstance(c, (list,tuple,np.ndarray))]
                if len(comps) == 0:
                    return None
                comps = np.array(comps, dtype=float)
                w = df['flow_rate'].astype(float).values
                return (w.reshape(-1,1) * comps).sum(axis=0) / (w.sum() + eps)

            pin = weighted_comp_mean(in_edges)
            pout = weighted_comp_mean(out_edges)
            
            if pin is None and pout is None:
                row['comp_aitchison'] = 0.0
                row['comp_js'] = 0.0
                row['comp_present'] = False
            else:
                row['comp_present'] = True
                if pin is None: pin = pout
                if pout is None: pout = pin
                
                try:
                    row['comp_aitchison'] = float(AspenNodeImportance_Calculator.aitchison_distance(pin, pout))
                except Exception:
                    row['comp_aitchison'] = 0.0
                try:
                    row['comp_js'] = float(AspenNodeImportance_Calculator.js_divergence(pin, pout))
                except Exception:
                    row['comp_js'] = 0.0

            rows.append(row)

        metrics_df = pd.DataFrame(rows)

        # Normalize all metric columns to [0,1]
        metric_cols = [c for c in metrics_df.columns if c != 'node_id']
        for c in metric_cols:
            arr = metrics_df[c].values.astype(float)
            lo, hi = np.nanmin(arr), np.nanmax(arr)
            if np.isclose(hi, lo):
                metrics_df[c + '_norm'] = 0.0
            else:
                metrics_df[c + '_norm'] = (arr - lo) / (hi - lo)

        # Combine scores per feature using weights
        for feat, wdict in feature_weights.items():
            score = np.zeros(len(metrics_df))  # initialize score array
            for metric_key, w in wdict.items():
                # Column naming logic: composition features use comp_aitchison/_js, others are like temperature_js
                if feat == 'composition':
                    colname = (metric_key + '_norm' if (metric_key + '_norm') in metrics_df.columns 
                              else metric_key)
                else:
                    colname = (feat + '_' + metric_key + '_norm' 
                              if (feat + '_' + metric_key + '_norm') in metrics_df.columns 
                              else feat + '_' + metric_key)
                if colname in metrics_df.columns:
                    score += w * metrics_df[colname].astype(float).values  # weighted sum
            
            # Normalize each feature score to [0,1]
            if np.ptp(score) == 0:  # if all scores are identical
                metrics_df[feat + '_score'] = 0.0
            else:
                metrics_df[feat + '_score'] = (score - score.min()) / score.ptp()  # range normalization

        # Ensure all feature score columns exist
        for c in ['flow_rate_score','temperature_score','pressure_score','composition_score']:
            if c not in metrics_df.columns:
                metrics_df[c] = 0.0
                
        # Create fused vector (list of all feature scores)
        metrics_df['fused_vector'] = metrics_df[['flow_rate_score','temperature_score',
                                                'pressure_score','composition_score']].values.tolist()
        # Combined score is the mean of feature scores
        metrics_df['combined_score'] = metrics_df[['flow_rate_score','temperature_score',
                                                  'pressure_score','composition_score']].mean(axis=1)

        return metrics_df

    # =============================
    # Convenience wrapper methods
    # =============================

    def run_pipeline(
        self,
        edges_df: pd.DataFrame,
        connections: List[Tuple[str,str,str]],
        edge_name_col: Optional[str] = None,
        pressure_cols_hint: Optional[List[str]] = None,
        temperature_cols_hint: Optional[List[str]] = None,
        flow_cols_hint: Optional[List[str]] = None,
        composition_cols_hint: Optional[List[str]] = None,
        skip_virtual: bool = True,
        is_virtual_fn: Optional[Callable[[str], bool]] = None,
        feature_weights: Optional[Dict[str, Dict[str, float]]] = None,
        bins: int = 20
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Full pipeline: transform edge data and compute node importance.
        
        Returns:
          (final_edges_df, metrics_df) - normalized edge DataFrame and node metrics DataFrame
        """
        # First step: transform edges
        final_edges = self.transform_edges_from_dataframe(
            edges_df, connections, edge_name_col=edge_name_col,
            pressure_cols_hint=pressure_cols_hint, temperature_cols_hint=temperature_cols_hint,
            flow_cols_hint=flow_cols_hint, composition_cols_hint=composition_cols_hint
        )
        
        # Second step: compute node importance
        metrics_df = self.compute_node_importance(
            final_edges, bins=bins, skip_virtual=skip_virtual,
            is_virtual_fn=is_virtual_fn, feature_weights=feature_weights
        )
        
        return final_edges, metrics_df


# =============================
# Example usage (if run as script)
# =============================
# if __name__ == '__main__':
#     # Create sample data
#     sample_edges = pd.DataFrame({
#         'edge': ['F01','F02','F03'],  # edge names
#         'Pressure/Bar': [1.0, 1.1, 0.9],  # pressure data
#         'Temperature/°C': [300, 310, 295],  # temperature data
#         'Massflow/(kg/h)': [100, 50, 20],  # mass flow data
#         'comp_list': ['[0.8,0.2]','[0.6,0.4]','[0.7,0.3]']  # composition data (string format)
#     })
    
#     # Define connections: each tuple is (edge_name, start_node, end_node)
#     sample_conn = [('F01','#1','V-1'), ('F02','V-1','P-1'), ('F03','P-1','REACT1')]

#     # Create pipeline instance and run
#     pip = AspenNodeImportance_Calculator()
#     fe, md = pip.run_pipeline(sample_edges, sample_conn, edge_name_col='edge')
    
#     # Print results
#     print("Final edges:\n", fe)
#     print("Metrics (nodes):\n", md)

