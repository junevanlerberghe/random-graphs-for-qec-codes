import sqlite3
from typing import Optional, Dict, Any, Iterable, Sequence, List, Tuple
import numpy as np


class GraphResultsDB:
    def __init__(self, db_path="database/qec_results.db"):
        self.db_path = db_path
        self.conn = sqlite3.connect(self.db_path)
        self.conn.execute("PRAGMA foreign_keys = ON;")
        self._create_schema()

    def _create_schema(self):
        cur = self.conn.cursor()

        # Table for each successful graph run (valid CSS code)
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS graph_runs (
                id               INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id           INTEGER,      -- script-level run counter
                model            TEXT,
                n_qubits         INTEGER,
                n_checks         INTEGER,
                p                REAL,
                m_ba             INTEGER,
                k_neighbors      INTEGER,
                repeat           INTEGER,
                edges            INTEGER,
                density_QxC      REAL,
                avg_degQ         REAL,
                min_degQ         INTEGER,
                max_degQ         INTEGER,
                avg_degC         REAL,
                min_degC         INTEGER,
                max_degC         INTEGER,
                is_connected     INTEGER,
                num_components   INTEGER,
                fiedler_lambda2  REAL,
                fiedler_vec_norm REAL,
                girth            REAL,
                logical_qubits   INTEGER,
                out_dir          TEXT
            )
            """
        )

        # Table for aggregated success rates per parameter setting
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS success_rates (
                id            INTEGER PRIMARY KEY AUTOINCREMENT,
                model         TEXT,
                n_qubits      INTEGER,
                n_checks      INTEGER,
                p             REAL,
                m_ba          INTEGER,
                k_neighbors   INTEGER,
                repeats       INTEGER,
                num_valid     INTEGER,
                success_rate  REAL
            )
            """
        )

        # Biadjacency matrix entries: one row per (q, c) cell
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS biadjacency (
                id             INTEGER PRIMARY KEY AUTOINCREMENT,
                graph_runs_id  INTEGER,   -- FK to graph_runs.id
                run_id         INTEGER,   -- script-level run_id
                row_label      TEXT,
                col_label      TEXT,
                value          INTEGER,
                FOREIGN KEY(graph_runs_id) REFERENCES graph_runs(id) ON DELETE CASCADE
            )
            """
        )
        cur.execute(
            "CREATE INDEX IF NOT EXISTS idx_biadj_run ON biadjacency(graph_runs_id)"
        )
        cur.execute(
            "CREATE INDEX IF NOT EXISTS idx_biadj_runid ON biadjacency(run_id)"
        )

        # Edge list: one row per edge
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS edges (
                id             INTEGER PRIMARY KEY AUTOINCREMENT,
                graph_runs_id  INTEGER,
                run_id         INTEGER,
                u              TEXT,
                v              TEXT,
                FOREIGN KEY(graph_runs_id) REFERENCES graph_runs(id) ON DELETE CASCADE
            )
            """
        )
        cur.execute("CREATE INDEX IF NOT EXISTS idx_edges_run ON edges(graph_runs_id)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_edges_runid ON edges(run_id)")

        # Degrees: one row per node
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS degrees (
                id             INTEGER PRIMARY KEY AUTOINCREMENT,
                graph_runs_id  INTEGER,
                run_id         INTEGER,
                node           TEXT,
                partition      TEXT,
                degree         INTEGER,
                FOREIGN KEY(graph_runs_id) REFERENCES graph_runs(id) ON DELETE CASCADE
            )
            """
        )
        cur.execute("CREATE INDEX IF NOT EXISTS idx_deg_run ON degrees(graph_runs_id)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_deg_runid ON degrees(run_id)")

        # Parity check matrix entries: one row per (i, j) cell in H (PCM)
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS parity_check_matrix (
                id             INTEGER PRIMARY KEY AUTOINCREMENT,
                graph_runs_id  INTEGER,
                run_id         INTEGER,
                row_index      INTEGER,
                col_index      INTEGER,
                value          INTEGER,
                FOREIGN KEY(graph_runs_id) REFERENCES graph_runs(id) ON DELETE CASCADE
            )
            """
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS code_metrics (
                id                        INTEGER PRIMARY KEY AUTOINCREMENT,
                graph_runs_id             INTEGER,
                run_id                    INTEGER,
                n_qubits                  INTEGER,
                k_logical                 INTEGER,
                distance                  INTEGER,
                code_rate                 REAL,
                degenerate                INTEGER,
                avg_logical_operator_weight REAL,
                avg_stabilizer_weight     REAL,
                stabilizer_wep            TEXT,
                normalizer_wep            TEXT,
                FOREIGN KEY(graph_runs_id) REFERENCES graph_runs(id) ON DELETE CASCADE
            )
            """
        )
        cur.execute(
            "CREATE INDEX IF NOT EXISTS idx_code_metrics_run ON code_metrics(graph_runs_id)"
        )

        
        cur.execute(
            "CREATE INDEX IF NOT EXISTS idx_pcm_run ON parity_check_matrix(graph_runs_id)"
        )
        cur.execute(
            "CREATE INDEX IF NOT EXISTS idx_pcm_runid ON parity_check_matrix(run_id)"
        )

        self.conn.commit()

    def insert_run(
        self,
        row,
        model,
        m_ba=None,
        k_neighbors=None,
    ):
        """
        Insert a single graph run (one valid CSS code instance) into graph_runs.
        Returns the auto-generated primary key id for this run (graph_runs.id),
        which should be passed as graph_runs_id to the other insert_* methods.
        """
        cur = self.conn.cursor()

        data = {
            "run_id":           row.get("run_id"),
            "model":            model,
            "n_qubits":         row.get("n_qubits"),
            "n_checks":         row.get("n_checks"),
            "p":                row.get("p"),
            "m_ba":             m_ba,
            "k_neighbors":      k_neighbors,
            "repeat":           row.get("repeat"),
            "edges":            row.get("edges"),
            "density_QxC":      row.get("density_QxC"),
            "avg_degQ":         row.get("avg_degQ"),
            "min_degQ":         row.get("min_degQ"),
            "max_degQ":         row.get("max_degQ"),
            "avg_degC":         row.get("avg_degC"),
            "min_degC":         row.get("min_degC"),
            "max_degC":         row.get("max_degC"),
            "is_connected":     int(row.get("is_connected")),
            "num_components":   row.get("num_components"),
            "fiedler_lambda2":  row.get("fiedler_lambda2"),
            "fiedler_vec_norm": row.get("fiedler_vec_norm"),
            "girth":            row.get("girth"),
            "logical_qubits":   row.get("logical_qubits"),
            "out_dir":          row.get("out_dir"),
        }

        cur.execute(
            """
            INSERT INTO graph_runs (
                run_id, model,
                n_qubits, n_checks, p, m_ba, k_neighbors, repeat,
                edges, density_QxC,
                avg_degQ, min_degQ, max_degQ,
                avg_degC, min_degC, max_degC,
                is_connected, num_components,
                fiedler_lambda2, fiedler_vec_norm,
                girth, logical_qubits, out_dir
            ) VALUES (
                :run_id, :model,
                :n_qubits, :n_checks, :p, :m_ba, :k_neighbors, :repeat,
                :edges, :density_QxC,
                :avg_degQ, :min_degQ, :max_degQ,
                :avg_degC, :min_degC, :max_degC,
                :is_connected, :num_components,
                :fiedler_lambda2, :fiedler_vec_norm,
                :girth, :logical_qubits, :out_dir
            )
            """,
            data,
        )
        self.conn.commit()
        # This is the DB primary key (graph_runs.id)
        return cur.lastrowid

    def insert_success_rate(
        self,
        *,
        model,
        n_qubits,
        n_checks,
        p,
        m_ba,
        k_neighbors,
        repeats,
        num_valid,
    ):
        cur = self.conn.cursor()
        success_rate = num_valid / repeats if repeats > 0 else 0.0

        data = {
            "model":        model,
            "n_qubits":     n_qubits,
            "n_checks":     n_checks,
            "p":            p,
            "m_ba":         m_ba,
            "k_neighbors":  k_neighbors,
            "repeats":      repeats,
            "num_valid":    num_valid,
            "success_rate": success_rate,
        }

        cur.execute(
            """
            INSERT INTO success_rates (
                model, n_qubits, n_checks, p, m_ba, k_neighbors,
                repeats, num_valid, success_rate
            ) VALUES (
                :model, :n_qubits, :n_checks, :p, :m_ba, :k_neighbors,
                :repeats, :num_valid, :success_rate
            )
            """,
            data,
        )
        self.conn.commit()

    def insert_biadjacency(
        self,
        graph_runs_id,
        run_id,
        Q,
        C,
        B,
    ):
        """
        Store full biadjacency matrix entries: one row per (q, c) cell.
        B is expected to be indexable as B[i, j].
        graph_runs_id: primary key from graph_runs.id
        run_id:        script-level run counter (same as in graph_runs.run_id)
        """
        cur = self.conn.cursor()
        records = []
        for i, q in enumerate(Q):
            for j, c in enumerate(C):
                value = int(B[i, j])
                records.append((graph_runs_id, run_id, q, c, value))

        cur.executemany(
            """
            INSERT INTO biadjacency (graph_runs_id, run_id, row_label, col_label, value)
            VALUES (?, ?, ?, ?, ?)
            """,
            records,
        )
        self.conn.commit()

    def insert_edges(
        self,
        graph_runs_id,
        run_id,
        edges,
    ):
        """
        Store edge list for a given run.
        edges should be an iterable of (u, v).
        """
        cur = self.conn.cursor()
        records = [
            (graph_runs_id, run_id, str(u), str(v))
            for (u, v) in edges
        ]
        cur.executemany(
            """
            INSERT INTO edges (graph_runs_id, run_id, u, v)
            VALUES (?, ?, ?, ?)
            """,
            records,
        )
        self.conn.commit()

    def insert_degrees(
        self,
        graph_runs_id,
        run_id,
        nodes,
        partitions,
        degrees,
    ):
        """
        Store node degrees for a given run.
        """
        cur = self.conn.cursor()
        records = [
            (graph_runs_id, run_id, str(node), part, int(deg))
            for node, part, deg in zip(nodes, partitions, degrees)
        ]
        cur.executemany(
            """
            INSERT INTO degrees (graph_runs_id, run_id, node, partition, degree)
            VALUES (?, ?, ?, ?, ?)
            """,
            records,
        )
        self.conn.commit()

    def insert_parity_check_matrix(
        self,
        graph_runs_id,
        run_id,
        pcm,
    ):
        """
        Store parity check matrix H (pcm) entry-wise for a given run.
        pcm is expected to be a 2D array-like object: pcm[i, j] in {0, 1}.
        """
        cur = self.conn.cursor()
        n_rows, n_cols = pcm.shape
        records = []
        for i in range(n_rows):
            for j in range(n_cols):
                value = int(pcm[i, j])
                records.append((graph_runs_id, run_id, i, j, value))

        cur.executemany(
            """
            INSERT INTO parity_check_matrix (
                graph_runs_id, run_id, row_index, col_index, value
            ) VALUES (?, ?, ?, ?, ?)
            """,
            records,
        )
        self.conn.commit()

    def insert_code_metrics(
        self,
        graph_runs_id,
        run_id,
        n_qubits,
        k_logical,
        distance,
        code_rate,
        degenerate,
        avg_logical_operator_weight,
        avg_stabilizer_weight,
        stabilizer_wep,
        normalizer_wep,
    ):
        cur = self.conn.cursor()

        data = {
            "graph_runs_id": graph_runs_id,
            "run_id": run_id,
            "n_qubits": n_qubits,
            "k_logical": k_logical,
            "distance": int(distance),
            "code_rate": float(code_rate),
            "degenerate": int(bool(degenerate)),
            "avg_logical_operator_weight": float(avg_logical_operator_weight),
            "avg_stabilizer_weight": float(avg_stabilizer_weight),
            "stabilizer_wep": str(stabilizer_wep),
            "normalizer_wep": str(normalizer_wep),
        }

        cur.execute(
            """
            INSERT INTO code_metrics (
                graph_runs_id,
                run_id,
                n_qubits,
                k_logical,
                distance,
                code_rate,
                degenerate,
                avg_logical_operator_weight,
                avg_stabilizer_weight,
                stabilizer_wep,
                normalizer_wep
            ) VALUES (
                :graph_runs_id,
                :run_id,
                :n_qubits,
                :k_logical,
                :distance,
                :code_rate,
                :degenerate,
                :avg_logical_operator_weight,
                :avg_stabilizer_weight,
                :stabilizer_wep,
                :normalizer_wep
            )
            """,
            data,
        )
        self.conn.commit()
        return cur.lastrowid


    def query_runs(
        self,
        model=None,
        p=None,
        n_qubits=None,
        n_checks=None,
        limit=None,
    ):
        sql = "SELECT * FROM graph_runs WHERE 1=1"
        params = {}

        if model is not None:
            sql += " AND model = :model"
            params["model"] = model
        if p is not None:
            sql += " AND ABS(p - :p) < 1e-12"
            params["p"] = p
        if n_qubits is not None:
            sql += " AND n_qubits = :n_qubits"
            params["n_qubits"] = n_qubits
        if n_checks is not None:
            sql += " AND n_checks = :n_checks"
            params["n_checks"] = n_checks

        sql += " ORDER BY id"

        if limit is not None:
            sql += " LIMIT :limit"
            params["limit"] = limit

        cur = self.conn.cursor()
        cur.execute(sql, params)
        rows = cur.fetchall()
        col_names = [d[0] for d in cur.description]

        return [dict(zip(col_names, row)) for row in rows]
    
    def get_graph_runs_ids_with_pcm(self):
        cur = self.conn.cursor()
        cur.execute(
            "SELECT DISTINCT graph_runs_id FROM parity_check_matrix"
        )
        rows = cur.fetchall()
        return [row[0] for row in rows]


    def get_run_by_id(self, graph_runs_id):
        cur = self.conn.cursor()
        cur.execute(
            "SELECT * FROM graph_runs WHERE id = ?",
            (graph_runs_id,),
        )
        row = cur.fetchone()
        if row is None:
            return None
        col_names = [d[0] for d in cur.description]
        return dict(zip(col_names, row))

    def get_biadjacency_for_run(self, graph_runs_id):
        cur = self.conn.cursor()
        cur.execute(
            """
            SELECT row_label, col_label, value
            FROM biadjacency
            WHERE graph_runs_id = ?
            """,
            (graph_runs_id,),
        )
        rows = cur.fetchall()
        if not rows:
            raise ValueError(f"No biadjacency entries for graph_runs_id={graph_runs_id}")

        # rows are tuples: (row_label, col_label, value)
        q_labels = sorted({row[0] for row in rows})
        c_labels = sorted({row[1] for row in rows})

        q_index = {q: i for i, q in enumerate(q_labels)}
        c_index = {c: j for j, c in enumerate(c_labels)}

        B = np.zeros((len(q_labels), len(c_labels)), dtype=int)
        for row in rows:
            i = q_index[row[0]]
            j = c_index[row[1]]
            B[i, j] = int(row[2])

        return q_labels, c_labels, B

    def get_parity_check_matrix_for_run(self, graph_runs_id):
        cur = self.conn.cursor()
        cur.execute(
            """
            SELECT row_index, col_index, value
            FROM parity_check_matrix
            WHERE graph_runs_id = ?
            """,
            (graph_runs_id,),
        )
        rows = cur.fetchall()
        if not rows:
            raise ValueError(f"No parity_check_matrix entries for graph_runs_id={graph_runs_id}")

        # rows are tuples: (row_index, col_index, value)
        max_row = max(row[0] for row in rows)
        max_col = max(row[1] for row in rows)

        pcm = np.zeros((max_row + 1, max_col + 1), dtype=int)
        for row in rows:
            i = row[0]
            j = row[1]
            pcm[i, j] = int(row[2])

        return pcm

    def get_runs_by_model_p(
        self,
        model,
        p,
        n_qubits=None,
        n_checks=None,
        limit=None,
    ):
        return self.query_runs(
            model=model,
            p=p,
            n_qubits=n_qubits,
            n_checks=n_checks,
            limit=limit,
        )

    def get_single_run_id_by_model_p(
        self,
        model,
        p,
        n_qubits=None,
        n_checks=None,
        index=0,
    ):
        runs = self.get_runs_by_model_p(
            model=model,
            p=p,
            n_qubits=n_qubits,
            n_checks=n_checks,
            limit=None,
        )
        if not runs:
            raise ValueError(
                f"No runs found for model={model}, p={p}, n_qubits={n_qubits}, n_checks={n_checks}"
            )
        if index >= len(runs):
            raise IndexError(
                f"index={index} out of range for {len(runs)} runs "
                f"(model={model}, p={p}, n_qubits={n_qubits}, n_checks={n_checks})"
            )
        return runs[index]["id"]

    def get_biadjacency_by_model_p(
        self,
        model,
        p,
        n_qubits=None,
        n_checks=None,
        index=0,
    ):
        graph_runs_id = self.get_single_run_id_by_model_p(
            model=model,
            p=p,
            n_qubits=n_qubits,
            n_checks=n_checks,
            index=index,
        )
        Q, C, B = self.get_biadjacency_for_run(graph_runs_id)
        return graph_runs_id, Q, C, B

    def get_parity_check_by_model_p(
        self,
        model,
        p,
        n_qubits=None,
        n_checks=None,
        index=0,
    ):
        graph_runs_id = self.get_single_run_id_by_model_p(
            model=model,
            p=p,
            n_qubits=n_qubits,
            n_checks=n_checks,
            index=index,
        )
        pcm = self.get_parity_check_matrix_for_run(graph_runs_id)
        return graph_runs_id, pcm
    
    def get_code_metrics_for_run(self, graph_runs_id):
        cur = self.conn.cursor()
        cur.execute(
            """
            SELECT
                id,
                graph_runs_id,
                run_id,
                n_qubits,
                k_logical,
                distance,
                code_rate,
                degenerate,
                avg_logical_operator_weight,
                avg_stabilizer_weight,
                stabilizer_wep,
                normalizer_wep
            FROM code_metrics
            WHERE graph_runs_id = ?
            """,
            (graph_runs_id,),
        )
        rows = cur.fetchall()
        if not rows:
            return []

        col_names = [d[0] for d in cur.description]
        return [dict(zip(col_names, row)) for row in rows]
    
    def get_code_metrics_by_model_p(
        self,
        model,
        p,
        n_qubits=None,
        n_checks=None,
    ):
        """
        Return a list of code_metrics rows joined with graph_runs,
        filtered by (model, p, optional n_qubits, n_checks).

        Each row is a dict with all fields from code_metrics plus
        model/p/n_qubits/n_checks from graph_runs.
        """
        sql = """
        SELECT
            cm.id AS code_metrics_id,
            cm.graph_runs_id,
            cm.run_id,
            cm.n_qubits,
            cm.k_logical,
            cm.distance,
            cm.code_rate,
            cm.degenerate,
            cm.avg_logical_operator_weight,
            cm.avg_stabilizer_weight,
            cm.stabilizer_wep,
            cm.normalizer_wep,
            gr.model,
            gr.p,
            gr.n_qubits AS run_n_qubits,
            gr.n_checks
        FROM code_metrics cm
        JOIN graph_runs gr
            ON cm.graph_runs_id = gr.id
        WHERE gr.model = :model
          AND ABS(gr.p - :p) < 1e-12
        """
        params = {"model": model, "p": p}

        if n_qubits is not None:
            sql += " AND gr.n_qubits = :n_qubits"
            params["n_qubits"] = n_qubits

        if n_checks is not None:
            sql += " AND gr.n_checks = :n_checks"
            params["n_checks"] = n_checks

        sql += " ORDER BY cm.id"

        cur = self.conn.cursor()
        cur.execute(sql, params)
        rows = cur.fetchall()
        if not rows:
            return []

        col_names = [d[0] for d in cur.description]
        return [dict(zip(col_names, row)) for row in rows]



    def close(self):
        self.conn.close()
