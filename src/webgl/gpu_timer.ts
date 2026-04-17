type PendingQuery = { query: WebGLQuery; label: string; frame: number };
export type GPUTimerResults = {[label: string]: number};

export class GPUTimer {
    private _gl: WebGL2RenderingContext;
    private _ext: any;
    private _pending: PendingQuery[] = [];
    private _results: GPUTimerResults = {};
    private _active: WebGLQuery | null = null;
    private _frame = 0;
    private _enabled: boolean;

    constructor(gl: WebGL2RenderingContext) {
        this._gl = gl;
        this._ext = gl.getExtension('EXT_disjoint_timer_query_webgl2');
        this._enabled = !!this._ext;
    }

    get enabled() { return this._enabled; }

    beginFrame() {
        if (!this._enabled) return;
        this._frame++;
        if (this._gl.getParameter(this._ext.GPU_DISJOINT_EXT)) {
            for (const p of this._pending) this._gl.deleteQuery(p.query);
            this._pending = [];
        }
    }

    start(label: string) {
        if (!this._enabled) return;
        if (this._active) { this._gl.endQuery(this._ext.TIME_ELAPSED_EXT); this._active = null; }
        const query = this._gl.createQuery();
        this._gl.beginQuery(this._ext.TIME_ELAPSED_EXT, query);
        this._active = query;
        this._pending.push({query, label, frame: this._frame});
    }

    end() {
        if (!this._enabled || !this._active) return;
        this._gl.endQuery(this._ext.TIME_ELAPSED_EXT);
        this._active = null;
    }

    collectResults(): GPUTimerResults {
        if (!this._enabled) return this._results;
        const gl = this._gl;
        const remaining: PendingQuery[] = [];
        for (const p of this._pending) {
            if (gl.getQueryParameter(p.query, gl.QUERY_RESULT_AVAILABLE)) {
                this._results[p.label] = gl.getQueryParameter(p.query, gl.QUERY_RESULT) / 1_000_000;
                gl.deleteQuery(p.query);
            } else if (this._frame - p.frame > 5) {
                gl.deleteQuery(p.query);
            } else {
                remaining.push(p);
            }
        }
        this._pending = remaining;
        return this._results;
    }

    destroy() {
        for (const p of this._pending) this._gl.deleteQuery(p.query);
        this._pending = [];
    }
}
