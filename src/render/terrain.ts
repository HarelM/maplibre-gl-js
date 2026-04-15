import {mat4, vec2} from 'gl-matrix';
import {OverscaledTileID} from '../tile/tile_id';
import {RGBAImage} from '../util/image';
import {warnOnce} from '../util/util';
import {Pos3dArray, TriangleIndexArray} from '../data/array_types.g';
import pos3dAttributes from '../data/pos3d_attributes';
import {SegmentVector} from '../data/segment';
import {Texture} from '../webgl/texture';
import {MercatorCoordinate} from '../geo/mercator_coordinate';
import {TerrainTileManager} from '../tile/terrain_tile_manager';
import {EXTENT} from '../data/extent';
import {earthRadius, type LngLat} from '../geo/lng_lat';
import {Mesh} from './mesh';
import {isInBoundsForZoomLngLat} from '../util/world_bounds';
import {NORTH_POLE_Y, SOUTH_POLE_Y} from './subdivision';
import {coveringTiles} from '../geo/projection/covering_tiles';
import type Point from '@mapbox/point-geometry';
import type {Tile} from '../tile/tile';
import type {Framebuffer} from '../webgl/framebuffer';
import type {TileManager} from '../tile/tile_manager';
import type {TerrainSpecification} from '@maplibre/maplibre-gl-style-spec';
import type {Painter} from './painter';
import type {IReadonlyTransform} from '../geo/transform_interface';

/**
 * @internal
 * A terrain GPU related object
 */
export type TerrainData = {
    'u_depth': number;
    'u_terrain': number;
    'u_terrain_dim': number;
    'u_terrain_matrix': mat4;
    'u_terrain_unpack': number[];
    'u_terrain_exaggeration': number;
    texture: WebGLTexture;
    depthTexture: WebGLTexture;
    tile: Tile;
};

/**
 * @internal
 * This is the main class which handles most of the 3D Terrain logic. It has the following topics:
 *
 * 1. loads raster-dem tiles via the internal tileManager this.tileManager
 * 2. creates a depth-framebuffer, which is used to calculate the visibility of coordinates
 * 3. creates a coords-framebuffer, which is used the get to tile-coordinate for a screen-pixel
 * 4. stores all render-to-texture tiles in the this.tileManager._tiles
 * 5. calculates the elevation for a specific tile-coordinate
 * 6. creates a terrain-mesh
 *
 * A note about the GPU resource-usage:
 *
 * Framebuffers:
 *
 * - one for the depth & coords framebuffer with the size of the map-div.
 * - one for rendering a tile to texture with the size of tileSize (= 512x512).
 *
 * Textures:
 *
 * - one texture for an empty raster-dem tile with size 1x1
 * - one texture for an empty depth-buffer, when terrain is disabled with size 1x1
 * - one texture for an each loaded raster-dem with size of the source.tileSize
 * - one texture for the coords-framebuffer with the size of the map-div.
 * - one texture for the depth-framebuffer with the size of the map-div.
 * - one texture for the encoded tile-coords with the size 2*tileSize (=1024x1024)
 * - finally for each render-to-texture tile (= this._tiles) a set of textures
 * for each render stack (The stack-concept is documented in painter.ts).
 *
 * Normally there exists 1-3 Textures per tile, depending on the stylesheet.
 * Each Textures has the size 2*tileSize (= 1024x1024). Also there exists a
 * cache of the last 150 newest rendered tiles.
 *
 */
export class Terrain {
    /**
     * The style this terrain corresponds to
     */
    painter: Painter;
    /**
     * the tilemanager this terrain is based on
     */
    tileManager: TerrainTileManager;
    /**
     * the TerrainSpecification object passed to this instance
     */
    options: TerrainSpecification;
    /**
     * define the meshSize per tile.
     */
    meshSize: number;
    /**
     * multiplicator for the elevation. Used to make terrain more "extreme".
     */
    exaggeration: number;
    /**
     * to not see pixels in the render-to-texture tiles it is good to render them bigger
     * this number is the multiplicator (must be a power of 2) for the current tileSize.
     * So to get good results with not too much memory footprint a value of 2 should be fine.
     */
    qualityFactor: number;
    /**
     * holds the framebuffer object in size of the screen to render the coords into a texture.
     */
    _fbo: Framebuffer;
    _fboCoordsTexture: Texture;
    /**
     * Separate framebuffer for the depth pass, using a native DEPTH_COMPONENT32F texture.
     */
    _depthFbo: Framebuffer;
    /**
     * Native WebGL2 depth texture (DEPTH_COMPONENT32F) attached to _depthFbo.
     * Sampled directly in shaders via u_depth.
     */
    _fboDepthTexture: WebGLTexture;
    /**
     * R32F color texture attached to _depthFbo for CPU readback in depthAtPoint().
     */
    _fboDepthColorTexture: WebGLTexture;
    _emptyDepthTexture: Texture;
    /** PBO for async depth readback. Avoids GPU pipeline stalls from synchronous readPixels. */
    _depthPbo: WebGLBuffer | null = null;
    _depthReadbackSync: WebGLSync | null = null;
    _depthReadbackResult: number = 0;
    _depthReadbackBuffer: Float32Array = new Float32Array(1);
    /**
     * GL Objects for the terrain-mesh
     * The mesh is a regular mesh, which has the advantage that it can be reused for all tiles.
     */
    _meshCache: { [key: string]: Mesh } = {};
    /**
     * coords index contains a list of tileID.keys. This index is used to identify
     * the tile via the alpha-cannel in the coords-texture.
     * As the alpha-channel has 1 Byte a max of 255 tiles can rendered without an error.
     */
    coordsIndex: string[];
    /**
     * tile-coords encoded in the rgb channel, _coordsIndex is in the alpha-channel.
     */
    _coordsTexture: Texture;
    /**
     * accuracy of the coords. 2 * tileSize should be enough.
     */
    _coordsTextureSize: number;
    /**
     * variables for an empty dem texture, which is used while the raster-dem tile is loading.
     */
    _emptyDemUnpack: number[];
    _emptyDemTexture: Texture;
    _emptyDemMatrix: mat4;
    /**
     * as of overzooming of raster-dem tiles in high zoomlevels, this cache contains
     * matrices to transform from vector-tile coords to raster-dem-tile coords.
     */
    _demMatrixCache: {[_: string]: { matrix: mat4; coord: OverscaledTileID }};

    constructor(painter: Painter, tileManager: TileManager, options: TerrainSpecification) {
        this.painter = painter;
        this.tileManager = new TerrainTileManager(tileManager);
        this.options = options;
        this.exaggeration = typeof options.exaggeration === 'number' ? options.exaggeration : 1.0;
        this.qualityFactor = 2;
        this.meshSize = 128;
        this._demMatrixCache = {};
        this.coordsIndex = [];
        this._coordsTextureSize = 1024;
    }

    destroy() {
        const gl = this.painter.context.gl as WebGL2RenderingContext;
        if (this._fbo) {
            this._fbo.destroy();
            this._fbo = null;
        }
        if (this._fboCoordsTexture) {
            this._fboCoordsTexture.destroy();
            this._fboCoordsTexture = null;
        }
        if (this._depthFbo) {
            this._depthFbo.destroy();
            this._depthFbo = null;
        }
        if (this._fboDepthTexture) {
            gl.deleteTexture(this._fboDepthTexture);
            this._fboDepthTexture = null;
        }
        if (this._fboDepthColorTexture) {
            gl.deleteTexture(this._fboDepthColorTexture);
            this._fboDepthColorTexture = null;
        }
        if (this._depthPbo) {
            gl.deleteBuffer(this._depthPbo);
            this._depthPbo = null;
        }
        if (this._depthReadbackSync) {
            gl.deleteSync(this._depthReadbackSync);
            this._depthReadbackSync = null;
        }
        if (this._emptyDemTexture) {
            this._emptyDemTexture.destroy();
            this._emptyDemTexture = null;
        }
        if (this._emptyDepthTexture) {
            this._emptyDepthTexture.destroy();
            this._emptyDepthTexture = null;
        }
        if (this._coordsTexture) {
            this._coordsTexture.destroy();
            this._coordsTexture = null;
        }
        for (const key in this._meshCache) {
            this._meshCache[key].destroy();
        }
        this._meshCache = {};
        this.tileManager.destruct();
    }

    /**
     * Get the elevation-value from original dem-data for a given tile-coordinate.
     * Coordinates that fall outside `[0, extent)` are normalized to the
     * appropriate neighbor tile before lookup.
     * @param tileID - the tile to get the elevation for
     * @param x - x coordinate relative to the tile, may be outside `[0, extent)`
     * @param y - y coordinate relative to the tile, may be outside `[0, extent)`
     * @param extent - optional, default 8192
     * @returns the elevation
     */
    getDEMElevation(tileID: OverscaledTileID, x: number, y: number, extent: number = EXTENT): number {
        const normalized = tileID.normalizeCoordinates(x, y, extent);
        if (!normalized) return 0;

        const terrain = this.getTerrainData(normalized.tileID);
        const dem = terrain.tile?.dem;
        if (!dem) return 0;

        const pos = vec2.transformMat4([] as any, [normalized.x / extent * EXTENT, normalized.y / extent * EXTENT], terrain.u_terrain_matrix);
        const coord = [pos[0] * dem.dim, pos[1] * dem.dim];

        // bilinear interpolation
        const cx = Math.floor(coord[0]),
            cy = Math.floor(coord[1]),
            tx = coord[0] - cx,
            ty = coord[1] - cy;
        return (
            dem.get(cx, cy) * (1 - tx) * (1 - ty) +
            dem.get(cx + 1, cy) * (tx) * (1 - ty) +
            dem.get(cx, cy + 1) * (1 - tx) * (ty) +
            dem.get(cx + 1, cy + 1) * (tx) * (ty)
        );
    }

    /**
     * Get the elevation for given {@link LngLat} in respect of exaggeration.
     * @param lnglat - the location
     * @param zoom - the zoom, use {@link getElevationForLngLat} if you don't want a specific zoom level, but more accurate results.
     * @returns the elevation
     */
    getElevationForLngLatZoom(lnglat: LngLat, zoom: number) {
        if (!isInBoundsForZoomLngLat(zoom, lnglat.wrap())) return 0;
        const {tileID, mercatorX, mercatorY} = this._getOverscaledTileIDFromLngLatZoom(lnglat, zoom);
        return this.getElevation(tileID, mercatorX % EXTENT, mercatorY % EXTENT, EXTENT);
    }

    /**
     * Get the elevation for given {@link LngLat} in respect of exaggeration.
     * This will traverse up the zoom levels to find the first tile with data to return.
     * @param lnglat - the location
     * @returns the elevation
     */
    getElevationForLngLat(lnglat: LngLat, transform: IReadonlyTransform) {
        const terrainCoveringTiles = coveringTiles(transform, {maxzoom: this.tileManager.maxzoom, minzoom: this.tileManager.minzoom, tileSize: 512, terrain: this});
        let zoom = 0;
        for (const tile of terrainCoveringTiles) {
            if (tile.canonical.z > zoom) {
                zoom = Math.min(tile.canonical.z, this.tileManager.maxzoom);
            }
        }
        return this.getElevationForLngLatZoom(lnglat, zoom);
    }

    /**
     * Get the elevation for given coordinate in respect of exaggeration.
     * @param tileID - the tile id
     * @param x - x coordinate relative to the tile, may be outside `[0, extent)`
     * @param y - y coordinate relative to the tile, may be outside `[0, extent)`
     * @param extent - optional, default 8192
     * @returns the elevation
     */
    getElevation(tileID: OverscaledTileID, x: number, y: number, extent: number = EXTENT): number {
        return this.getDEMElevation(tileID, x, y, extent) * this.exaggeration;
    }

    /**
     * returns a Terrain Object for a tile. Unless the tile corresponds to data (e.g. tile is loading), return a flat dem object
     * @param tileID - the tile to get the terrain for
     * @returns the terrain data to use in the program
     */
    getTerrainData(tileID: OverscaledTileID): TerrainData {
        // create empty DEM Objects, which will used while raster-dem tiles are loading.
        // creates an empty depth-buffer texture which is needed, during the initialization process of the 3d mesh..
        if (!this._emptyDemTexture) {
            const context = this.painter.context;
            const image = new RGBAImage({width: 1, height: 1}, new Uint8Array(1 * 4));
            this._emptyDepthTexture = new Texture(context, image, context.gl.RGBA, {premultiply: false});
            this._emptyDemUnpack = [0, 0, 0, 0];
            this._emptyDemTexture = new Texture(context, new RGBAImage({width: 1, height: 1}), context.gl.RGBA, {premultiply: false});
            this._emptyDemTexture.bind(context.gl.NEAREST, context.gl.CLAMP_TO_EDGE);
            this._emptyDemMatrix = mat4.identity([] as any);
        }
        // find covering dem tile and prepare demTexture
        const sourceTile = this.tileManager.getSourceTile(tileID, true);
        if (sourceTile?.dem && (!sourceTile.demTexture || sourceTile.needsTerrainPrepare)) {
            const context = this.painter.context;
            sourceTile.demTexture = this.painter.getTileTexture(sourceTile.dem.stride);
            if (sourceTile.demTexture) sourceTile.demTexture.update(sourceTile.dem.getPixels(), {premultiply: false});
            else sourceTile.demTexture = new Texture(context, sourceTile.dem.getPixels(), context.gl.RGBA, {premultiply: false});
            sourceTile.demTexture.bind(context.gl.NEAREST, context.gl.CLAMP_TO_EDGE);
            sourceTile.needsTerrainPrepare = false;
        }
        // create matrix for lookup in dem data
        const matrixKey = sourceTile && sourceTile.toString() + sourceTile.tileID.key + tileID.key;
        if (matrixKey && !this._demMatrixCache[matrixKey]) {
            const maxzoom = this.tileManager.getSource().maxzoom;
            let dz = tileID.canonical.z - sourceTile.tileID.canonical.z;
            if (tileID.overscaledZ > tileID.canonical.z) {
                if (tileID.canonical.z >= maxzoom) dz =  tileID.canonical.z - maxzoom;
                else warnOnce('cannot calculate elevation if elevation maxzoom > source.maxzoom');
            }
            const dx = tileID.canonical.x - (tileID.canonical.x >> dz << dz);
            const dy = tileID.canonical.y - (tileID.canonical.y >> dz << dz);
            const demMatrix = mat4.fromScaling(new Float64Array(16) as any, [1 / (EXTENT << dz), 1 / (EXTENT << dz), 0]);
            mat4.translate(demMatrix, demMatrix, [dx * EXTENT, dy * EXTENT, 0]);
            this._demMatrixCache[matrixKey] = {matrix: demMatrix, coord: tileID};
        }
        // return uniform values & textures
        return {
            'u_depth': 2,
            'u_terrain': 3,
            'u_terrain_dim': sourceTile?.dem?.dim || 1,
            'u_terrain_matrix': matrixKey ? this._demMatrixCache[matrixKey].matrix : this._emptyDemMatrix,
            'u_terrain_unpack': sourceTile?.dem?.getUnpackVector() || this._emptyDemUnpack,
            'u_terrain_exaggeration': this.exaggeration,
            texture: (sourceTile?.demTexture || this._emptyDemTexture).texture,
            depthTexture: this._fboDepthTexture || this._emptyDepthTexture.texture,
            tile: sourceTile
        };
    }

    /**
     * get a framebuffer as big as the map-div, which will be used to render depth & coords into a texture
     * @param texture - the texture
     * @returns the frame buffer
     */
    getFramebuffer(texture: 'depth' | 'coords'): Framebuffer {
        const painter = this.painter;
        const width = painter.width / devicePixelRatio;
        const height = painter.height / devicePixelRatio;

        if (texture === 'depth') {
            return this._getDepthFramebuffer(width, height);
        }

        // Coords framebuffer
        if (this._fbo && (this._fbo.width !== width || this._fbo.height !== height)) {
            this._fbo.destroy();
            this._fboCoordsTexture.destroy();
            delete this._fbo;
            delete this._fboCoordsTexture;
        }
        if (!this._fboCoordsTexture) {
            this._fboCoordsTexture = new Texture(painter.context, {width, height, data: null}, painter.context.gl.RGBA, {premultiply: false});
            this._fboCoordsTexture.bind(painter.context.gl.NEAREST, painter.context.gl.CLAMP_TO_EDGE);
        }
        if (!this._fbo) {
            this._fbo = painter.context.createFramebuffer(width, height, true, false);
            this._fbo.depthAttachment.set(painter.context.createRenderbuffer(painter.context.gl.DEPTH_COMPONENT16, width, height));
        }
        this._fbo.colorAttachment.set(this._fboCoordsTexture.texture);
        return this._fbo;
    }

    /**
     * Create and return the depth framebuffer using native WebGL2 depth textures.
     * Uses DEPTH_COMPONENT32F for the depth attachment (sampled in shaders)
     * and R32F for the color attachment (used for CPU readback in depthAtPoint).
     */
    private _getDepthFramebuffer(width: number, height: number): Framebuffer {
        const painter = this.painter;
        const gl = painter.context.gl as WebGL2RenderingContext;

        // Destroy and recreate if size changed
        if (this._depthFbo && (this._depthFbo.width !== width || this._depthFbo.height !== height)) {
            this._depthFbo.destroy();
            gl.deleteTexture(this._fboDepthTexture);
            gl.deleteTexture(this._fboDepthColorTexture);
            this._depthFbo = null;
            this._fboDepthTexture = null;
            this._fboDepthColorTexture = null;
        }

        if (!this._depthFbo) {
            // Create native DEPTH_COMPONENT32F depth texture
            const depthTexture = gl.createTexture();
            gl.bindTexture(gl.TEXTURE_2D, depthTexture);
            gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
            gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
            gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
            gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
            gl.texImage2D(gl.TEXTURE_2D, 0, gl.DEPTH_COMPONENT32F, width, height, 0, gl.DEPTH_COMPONENT, gl.FLOAT, null);
            this._fboDepthTexture = depthTexture;

            // Create R32F color texture for CPU readback
            const colorTexture = gl.createTexture();
            gl.bindTexture(gl.TEXTURE_2D, colorTexture);
            gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
            gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
            gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
            gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
            gl.texImage2D(gl.TEXTURE_2D, 0, gl.R32F, width, height, 0, gl.RED, gl.FLOAT, null);
            this._fboDepthColorTexture = colorTexture;

            // Create FBO without built-in depth (we attach manually)
            const fbo = gl.createFramebuffer();
            painter.context.bindFramebuffer.set(fbo);

            // Attach depth texture
            gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.DEPTH_ATTACHMENT, gl.TEXTURE_2D, depthTexture, 0);

            // Attach R32F color texture
            gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, colorTexture, 0);

            painter.context.bindFramebuffer.set(null);

            // Wrap in a Framebuffer-like object
            this._depthFbo = {
                framebuffer: fbo,
                width,
                height,
                colorAttachment: {
                    get: () => colorTexture,
                    set: () => { /* fixed attachment */ },
                    setDirty: () => { /* no-op */ },
                },
                depthAttachment: {
                    get: () => null,
                    set: () => { /* depth is a texture, not a renderbuffer */ },
                },
                destroy: () => {
                    gl.deleteFramebuffer(fbo);
                },
            } as any as Framebuffer;
        }

        return this._depthFbo;
    }

    /**
     * create coords texture, needed to grab coordinates from canvas
     * encode coords coordinate into 4 bytes:
     *   - 8 lower bits for x
     *   - 8 lower bits for y
     *   - 4 higher bits for x
     *   - 4 higher bits for y
     *   - 8 bits for coordsIndex (1 .. 255) (= number of terraintile), is later set in draw_terrain uniform value
     * @returns the texture
     */
    getCoordsTexture(): Texture {
        const context = this.painter.context;
        if (this._coordsTexture) return this._coordsTexture;
        const data = new Uint8Array(this._coordsTextureSize * this._coordsTextureSize * 4);
        for (let y = 0, i = 0; y < this._coordsTextureSize; y++) for (let x = 0; x < this._coordsTextureSize; x++, i += 4) {
            data[i + 0] = x & 255;
            data[i + 1] = y & 255;
            data[i + 2] = ((x >> 8) << 4) | (y >> 8);
            data[i + 3] = 0;
        }
        const image = new RGBAImage({width: this._coordsTextureSize, height: this._coordsTextureSize}, new Uint8Array(data.buffer));
        const texture = new Texture(context, image, context.gl.RGBA, {premultiply: false});
        texture.bind(context.gl.NEAREST, context.gl.CLAMP_TO_EDGE);
        this._coordsTexture = texture;
        return texture;
    }

    /**
     * Reads a pixel from the coords-framebuffer and translate this to mercator, or null, if the pixel doesn't lie on the terrain's surface (but the sky instead).
     * @param p - Screen-Coordinate
     * @returns Mercator coordinate for a screen pixel, or null, if the pixel is not covered by terrain (is in the sky).
     */
    pointCoordinate(p: Point): MercatorCoordinate {
        // First, ensure the coords framebuffer is up to date.
        this.painter.maybeDrawDepth(true);
        this.painter.maybeDrawCoords();

        const rgba = new Uint8Array(4);
        const context = this.painter.context, gl = context.gl;
        const px = Math.round(p.x * this.painter.pixelRatio / devicePixelRatio);
        const py = Math.round(p.y * this.painter.pixelRatio / devicePixelRatio);
        const fbHeight = Math.round(this.painter.height / devicePixelRatio);
        // grab coordinate pixel from coordinates framebuffer
        context.bindFramebuffer.set(this.getFramebuffer('coords').framebuffer);
        gl.readPixels(px, fbHeight - py - 1, 1, 1, gl.RGBA, gl.UNSIGNED_BYTE, rgba);
        context.bindFramebuffer.set(null);
        // decode coordinates (encoding see getCoordsTexture)
        const x = rgba[0] + ((rgba[2] >> 4) << 8);
        const y = rgba[1] + ((rgba[2] & 15) << 8);
        const tileID = this.coordsIndex[255 - rgba[3]];
        const tile = tileID && this.tileManager.getTileByID(tileID);

        if (!tile) {
            return null;
        }

        const coordsSize = this._coordsTextureSize;
        const worldSize = (1 << tile.tileID.canonical.z) * coordsSize;
        return new MercatorCoordinate(
            (tile.tileID.canonical.x * coordsSize + x) / worldSize + tile.tileID.wrap,
            (tile.tileID.canonical.y * coordsSize + y) / worldSize,
            this.getElevation(tile.tileID, x, y, coordsSize)
        );
    }

    /**
     * Reads the depth value from the depth-framebuffer at a given screen pixel.
     * Uses PBO + fenceSync for async readback: returns the previous frame's result
     * immediately while issuing a new non-blocking read for the current frame.
     * The 1-frame latency is acceptable since marker occlusion is throttled to 100ms.
     * @param p - Screen coordinate
     * @returns depth value in NDC space (gl_Position.z / gl_Position.w)
     */
    depthAtPoint(p: Point): number {
        const context = this.painter.context;
        const gl = context.gl as WebGL2RenderingContext;

        // Collect previous async readback if ready
        if (this._depthReadbackSync) {
            const status = gl.clientWaitSync(this._depthReadbackSync, 0, 0);
            if (status === gl.ALREADY_SIGNALED || status === gl.CONDITION_SATISFIED) {
                gl.bindBuffer(gl.PIXEL_PACK_BUFFER, this._depthPbo);
                gl.getBufferSubData(gl.PIXEL_PACK_BUFFER, 0, this._depthReadbackBuffer);
                gl.bindBuffer(gl.PIXEL_PACK_BUFFER, null);
                this._depthReadbackResult = this._depthReadbackBuffer[0] * 2.0 - 1.0;
                gl.deleteSync(this._depthReadbackSync);
                this._depthReadbackSync = null;
            }
        }

        // Issue new async readback via PBO
        if (!this._depthPbo) {
            this._depthPbo = gl.createBuffer();
            gl.bindBuffer(gl.PIXEL_PACK_BUFFER, this._depthPbo);
            gl.bufferData(gl.PIXEL_PACK_BUFFER, 4, gl.STREAM_READ);
            gl.bindBuffer(gl.PIXEL_PACK_BUFFER, null);
        }

        context.bindFramebuffer.set(this.getFramebuffer('depth').framebuffer);
        gl.bindBuffer(gl.PIXEL_PACK_BUFFER, this._depthPbo);
        gl.readPixels(p.x, this.painter.height / devicePixelRatio - p.y - 1, 1, 1, gl.RED, gl.FLOAT, 0);
        gl.bindBuffer(gl.PIXEL_PACK_BUFFER, null);
        context.bindFramebuffer.set(null);

        if (this._depthReadbackSync) {
            gl.deleteSync(this._depthReadbackSync);
        }
        this._depthReadbackSync = gl.fenceSync(gl.SYNC_GPU_COMMANDS_COMPLETE, 0);
        gl.flush();

        return this._depthReadbackResult;
    }

    /**
     * create a regular mesh which will be used by all terrain-tiles
     * @returns the created regular mesh
     */
    getTerrainMesh(tileId: OverscaledTileID): Mesh {
        const globeEnabled = this.painter.style.projection?.transitionState > 0;
        const northPole = globeEnabled && tileId.canonical.y === 0;
        const southPole = globeEnabled && tileId.canonical.y === (1 << tileId.canonical.z) - 1;
        const key = `m_${northPole ? 'n' : ''}_${southPole ? 's' : ''}`;
        if (this._meshCache[key]) {
            return this._meshCache[key];
        }
        const context = this.painter.context;

        const vertexArray = new Pos3dArray();
        const indexArray = new TriangleIndexArray();
        const meshSize = this.meshSize;
        const delta = EXTENT / meshSize;
        const meshSize2 = meshSize * meshSize;
        for (let y = 0; y <= meshSize; y++) for (let x = 0; x <= meshSize; x++) {
            vertexArray.emplaceBack(x * delta, y * delta, 0);
        }
        for (let y = 0; y < meshSize2; y += meshSize + 1) for (let x = 0; x < meshSize; x++) {
            indexArray.emplaceBack(x + y, meshSize + x + y + 1, meshSize + x + y + 2);
            indexArray.emplaceBack(x + y, meshSize + x + y + 2, x + y + 1);
        }
        // add an extra frame around the mesh to avoid stitching on tile boundaries with different zoomlevels
        // top-bottom frame + pole vertices, if needed
        const offsetTop = vertexArray.length;
        const offsetTopEdge = 0;
        const offsetBottom = offsetTop + (meshSize + 1);
        const offsetBottomEdge = (meshSize + 1) * meshSize;
        const northY = northPole ? NORTH_POLE_Y : 0;
        const northZ = northPole ? 0 : 1;
        const southY = southPole ? SOUTH_POLE_Y : EXTENT;
        const southZ = southPole ? 0 : 1;
        for (let x = 0; x <= meshSize; x++) {
            vertexArray.emplaceBack(x * delta, northY, northZ);
        }
        for (let x = 0; x <= meshSize; x++) {
            vertexArray.emplaceBack(x * delta, southY, southZ);
        }
        for (let x = 0; x < meshSize; x++) {
            indexArray.emplaceBack(offsetBottomEdge + x, offsetBottom + x, offsetBottom + x + 1);
            indexArray.emplaceBack(offsetBottomEdge + x, offsetBottom + x + 1, offsetBottomEdge + x + 1);
            indexArray.emplaceBack(offsetTopEdge + x, offsetTop + x + 1, offsetTop + x);
            indexArray.emplaceBack(offsetTopEdge + x, offsetTopEdge + x + 1, offsetTop + x + 1);
        }
        // left-right frame
        const offsetLeft = vertexArray.length;
        const offsetRight = offsetLeft + (meshSize + 1) * 2;
        for (const x of [0, 1]) for (let y = 0; y <= meshSize; y++) for (const z of [0, 1]) {
            vertexArray.emplaceBack(x * EXTENT, y * delta, z);
        }
        for (let y = 0; y < meshSize * 2; y += 2) {
            indexArray.emplaceBack(offsetLeft + y, offsetLeft + y + 1, offsetLeft + y + 3);
            indexArray.emplaceBack(offsetLeft + y, offsetLeft + y + 3, offsetLeft + y + 2);
            indexArray.emplaceBack(offsetRight + y, offsetRight + y + 3, offsetRight + y + 1);
            indexArray.emplaceBack(offsetRight + y, offsetRight + y + 2, offsetRight + y + 3);
        }

        const mesh = new Mesh(
            context.createVertexBuffer(vertexArray, pos3dAttributes.members),
            context.createIndexBuffer(indexArray),
            SegmentVector.simpleSegment(0, 0, vertexArray.length, indexArray.length)
        );
        this._meshCache[key] = mesh;
        return mesh;
    }

    /**
     * Calculates a height of the frame around the terrain-mesh to avoid stitching between
     * tile boundaries in different zoomlevels.
     * @param zoom - current zoomlevel
     * @returns the elevation delta in meters
     */
    getMeshFrameDelta(zoom: number): number {
        // divide by 5 is evaluated by trial & error to get a frame in the right height
        return 2 * Math.PI * earthRadius / Math.pow(2, Math.max(zoom, 0)) / 5;
    }

    getMinTileElevationForLngLatZoom(lnglat: LngLat, zoom: number) {
        if (!isInBoundsForZoomLngLat(zoom, lnglat.wrap())) return 0;
        const {tileID} = this._getOverscaledTileIDFromLngLatZoom(lnglat, zoom);
        return this.getMinMaxElevation(tileID).minElevation ?? 0;
    }

    /**
     * Get the minimum and maximum elevation contained in a tile. This includes any
     * exaggeration included in the terrain.
     *
     * @param tileID - ID of the tile to be used as a source for the min/max elevation
     * @returns the minimum and maximum elevation found in the tile, including the terrain's
     * exaggeration
     */
    getMinMaxElevation(tileID: OverscaledTileID): {minElevation: number | null; maxElevation: number | null} {
        const tile = this.getTerrainData(tileID).tile;
        const minMax = {minElevation: null, maxElevation: null};
        if (tile?.dem) {
            minMax.minElevation = tile.dem.min * this.exaggeration;
            minMax.maxElevation = tile.dem.max * this.exaggeration;
        }
        return minMax;
    }

    _getOverscaledTileIDFromLngLatZoom(lnglat: LngLat, zoom: number): { tileID: OverscaledTileID; mercatorX: number; mercatorY: number} {
        const mercatorCoordinate = MercatorCoordinate.fromLngLat(lnglat.wrap());
        const worldSize = (1 << zoom) * EXTENT;
        const mercatorX = mercatorCoordinate.x * worldSize;
        const mercatorY = mercatorCoordinate.y * worldSize;
        const tileX = Math.floor(mercatorX / EXTENT), tileY = Math.floor(mercatorY / EXTENT);
        const tileID = new OverscaledTileID(zoom, 0, zoom, tileX, tileY);
        return {
            tileID,
            mercatorX,
            mercatorY
        };
    }
}
