//
//  Shaders.metal
//  MetalCanvasTest
//
//  Compute shaders for GPU-accelerated content generators
//

#include <metal_stdlib>
#include <simd/simd.h>

using namespace metal;

constant float PI = 3.14159265;

struct UniversalVisualizerParams {
    float time;
    float2 outputSize;
    float param0;
    float param1;
    float param2;
    float param3;
    float param4;
    float param5;
};

// MARK: - Blue Fragments Generator
static float bf_sdSphere(float3 pos, float size) { return length(pos) - size; }
static float bf_sdBox(float3 pos, float3 size) { pos = abs(pos) - size; return max(max(pos.x, pos.y), pos.z); }
static float bf_sdOctahedron(float3 p, float s) {
    p = abs(p); float m = p.x + p.y + p.z - s; float3 q;
    if (3.0 * p.x < m) q = p.xyz; else if (3.0 * p.y < m) q = p.yzx; else if (3.0 * p.z < m) q = p.zxy; else return m * 0.57735027;
    float k = clamp(0.5 * (q.z - q.y + s), 0.0, s); return length(float3(q.x, q.y - s + k, q.z - k));
}
static float2x2 bf_rotate(float a) { float s = sin(a), c = cos(a); return float2x2(float2(c, s), float2(-s, c)); }
static float3 bf_repeat(float3 pos, float3 span) { float3 m = pos - span * floor(pos / span); return abs(m) - span * 0.5; }
static float bf_getDistance(float3 pos, float2 uv, float iTime) {
    float3 originalPos = pos;
    for (int i = 0; i < 3; i++) { pos = abs(pos) - 4.5; pos.xz = bf_rotate(1.0) * pos.xz; pos.yz = bf_rotate(1.0) * pos.yz; }
    pos = bf_repeat(pos, float3(4.0));
    float d0 = abs(originalPos.x) - 0.1;
    float d1 = bf_sdBox(pos, float3(0.8));
    pos.xy = bf_rotate(mix(1.0, 2.0, abs(sin(iTime)))) * pos.xy;
    float size = mix(1.1, 1.3, abs(uv.y) * abs(uv.x));
    float d2 = bf_sdSphere(pos, size);
    float dd2 = bf_sdOctahedron(pos, 1.8);
    float ddd2 = mix(d2, dd2, abs(sin(iTime)));
    return max(max(d1, -ddd2), -d0);
}

kernel void blueFragmentsKernel(texture2d<float, access::write> output [[texture(0)]], constant UniversalVisualizerParams& params [[buffer(0)]], uint2 gid [[thread_position_in_grid]]) {
    if (gid.x >= uint(params.outputSize.x) || gid.y >= uint(params.outputSize.y)) return;
    float2 res = params.outputSize; float iTime = params.time * params.param0; float2 p = (float2(gid) * 2.0 - res) / min(res.x, res.y);
    float3 cameraOrigin = float3(0.0, 0.0, -10.0 + iTime * 4.0);
    float3 cameraTarget = float3(cos(iTime) + sin(iTime / 2.0) * 10.0, exp(sin(iTime)) * 2.0, 3.0 + iTime * 4.0);
    float3 upDirection = float3(0.0, 1.0, 0.0); float3 cameraDir = normalize(cameraTarget - cameraOrigin);
    float3 cameraRight = normalize(cross(upDirection, cameraOrigin)); float3 cameraUp = cross(cameraDir, cameraRight);
    float3 rayDirection = normalize(cameraRight * p.x + cameraUp * p.y + cameraDir);
    float depth = 0.0; float ac = 0.0; float d = 0.0;
    for (int i = 0; i < 80; i++) {
        float3 rayPos = cameraOrigin + rayDirection * depth; d = bf_getDistance(rayPos, p, iTime);
        if (abs(d) < 0.0001) break;
        ac += exp(-d * mix(5.0, 10.0, abs(sin(iTime)))); depth += d;
    }
    float3 col = float3(0.0, 0.3, 0.7); ac *= 1.2 * (res.x / res.y - abs(p.x)); float3 finalCol = col * ac * 0.06;
    output.write(float4(finalCol, 1.0), gid);
}

// MARK: - Solid Color Generator
kernel void solidColorGeneratorKernel(texture2d<float, access::write> output [[texture(0)]], constant UniversalVisualizerParams& params [[buffer(0)]], uint2 gid [[thread_position_in_grid]]) {
    float4 color = float4(params.param0, params.param1, params.param2, params.param3);
    output.write(color, gid);
}

// MARK: - Plasma Generator
kernel void plasmaGeneratorKernel(texture2d<float, access::write> output [[texture(0)]], constant UniversalVisualizerParams& params [[buffer(0)]], uint2 gid [[thread_position_in_grid]]) {
    if (gid.x >= uint(params.outputSize.x) || gid.y >= uint(params.outputSize.y)) return;
    float2 uv = float2(gid) / params.outputSize; float t = params.time * params.param0;
    float v1 = sin(uv.x * 10.0 + t); float v2 = sin(10.0 * (uv.x * sin(t / 2.0) + uv.y * cos(t / 3.0)) + t);
    float v3 = sin(length(uv - float2(0.5 + 0.5 * sin(t / 5.0), 0.5 + 0.5 * cos(t / 3.0))) * 10.0);
    float v4 = sin(length(uv - float2(0.5 * cos(t / 2.0), 0.5 * sin(t / 4.0))) * 8.0);
    float value = (v1 + v2 + v3 + v4) / 4.0;
    float r = sin(value * M_PI_F) * 0.5 + 0.5; float g = sin(value * M_PI_F + 2.094) * 0.5 + 0.5; float b = sin(value * M_PI_F + 4.189) * 0.5 + 0.5;
    output.write(float4(r, g, b, 1.0), gid);
}

// MARK: - Color Wheel Generator
kernel void colorWheelGeneratorKernel(texture2d<float, access::write> output [[texture(0)]], constant UniversalVisualizerParams& params [[buffer(0)]], uint2 gid [[thread_position_in_grid]]) {
    if (gid.x >= uint(params.outputSize.x) || gid.y >= uint(params.outputSize.y)) return;
    float2 uv = float2(gid) / params.outputSize; float2 fromCenter = uv - float2(0.5, 0.5);
    float angle = atan2(fromCenter.y, fromCenter.x) + params.time * params.param0; float hue = fract((angle / (2.0 * M_PI_F)) + 0.5);
    float3 rgb; float h = hue * 6.0; float c = 1.0; float x = c * (1.0 - abs(fmod(h, 2.0) - 1.0));
    if (h < 1.0) rgb = float3(c, x, 0); else if (h < 2.0) rgb = float3(x, c, 0); else if (h < 3.0) rgb = float3(0, c, x);
    else if (h < 4.0) rgb = float3(0, x, c); else if (h < 5.0) rgb = float3(x, 0, c); else rgb = float3(c, 0, x);
    float alpha = smoothstep(1.0, 0.9, length(fromCenter) * 2.0);
    output.write(float4(rgb, alpha), gid);
}

// MARK: - Psychedelics GENERATOR
kernel void psychedelicsGeneratorKernel(texture2d<float, access::write> output [[texture(0)]], constant UniversalVisualizerParams& params [[buffer(0)]], uint2 gid [[thread_position_in_grid]]) {
    float width = params.outputSize.x; float height = params.outputSize.y;
    if (gid.x >= uint(width) || gid.y >= uint(height)) return;
    float2 uv = (float2(gid) - float2(width * 0.5, height * 0.5)) / min(width, height);
    float time = params.time * params.param0; // param0 represents intensity
    float2 cir = uv * uv + sin(uv.x * 15.0 + time) / 15.0 * sin(uv.y * 7.0 + time * 1.75) / 2.0 + uv.x * sin(time) / 16.0 + uv.y * sin(time * 1.25) / 16.0;
    float circles = sqrt(abs(cir.x + cir.y * 0.5) * 35.0) * 5.0;
    float3 col; col.r = sin(circles * 1.25 + 2.0); col.g = abs(sin(circles - 1.0) - sin(circles)); col.b = abs(sin(circles));
    output.write(float4(col, 1.0), gid);
}

// MARK: - Lightspeed GENERATOR
kernel void lightspeedGeneratorKernel(texture2d<float, access::write> output [[texture(0)]], constant UniversalVisualizerParams& params [[buffer(0)]], uint2 gid [[thread_position_in_grid]]) {
    float width = params.outputSize.x; float height = params.outputSize.y;
    if (gid.x >= uint(width) || gid.y >= uint(height)) return;
    float2 suv = (float2(gid) - float2(width * 0.5, height * 0.5)) / height;
    float2 uv = float2(length(suv), atan2(suv.y, suv.x)); float time = -params.time;
    float raysn = params.param0; float fx = params.param1; float fy = params.param2;
    float offset = 0.1 * sin(uv.y * 10.0 - time * 0.6) * cos(uv.y * 48.0 + time * 0.3) * cos(uv.y * 3.7 + time);
    float rayIntensity = (sin(uv.y * raysn + time) * 0.5 + 0.5) * (sin(uv.y * fx - time * 0.6) * 0.5 + 0.5) *
                         (sin(uv.y * fy + time * 0.8) * 0.5 + 0.5) * (1.0 - cos(uv.y + 22.0 * time - pow(uv.x + offset, 0.3) * 60.0)) * (uv.x * 2.0);
    output.write(float4(rayIntensity * 0.7, rayIntensity * 0.7, rayIntensity, 1.0), gid);
}

// MARK: - Fractal Pattern GENERATOR
inline float3 fractalPalette(float t) {
    float3 a = float3(0.5, 0.5, 0.5); float3 b = float3(0.5, 0.5, 0.5); float3 c = float3(1.0, 1.0, 1.0); float3 d = float3(0.263, 0.416, 0.557);
    return a + b * cos(6.28318 * (c * t * d));
}
kernel void fractalPatternGeneratorKernel(texture2d<float, access::write> output [[texture(0)]], constant UniversalVisualizerParams& params [[buffer(0)]], uint2 gid [[thread_position_in_grid]]) {
    float width = params.outputSize.x; float height = params.outputSize.y;
    if (gid.x >= uint(width) || gid.y >= uint(height)) return;
    float2 uv = float2(gid) / float2(width, height); uv = (uv - 0.5) * 2.0; uv.x *= (width / height);
    float3 finalColor = float3(0.0); float2 uv0 = uv;
    int iterations = int(params.param0); float repeatness = params.param1; float phase = params.param2;
    for (int i = 0; i < iterations; i++) {
        uv = fract(uv * repeatness) - 0.5; float d = length(uv) * exp(-length(uv0));
        float3 col = fractalPalette(length(uv0) + float(i) * 0.4 + params.time * 0.4);
        d = sin(d * phase + params.time) / 8.0; d = abs(d); d = pow(0.01 / d, 1.2); finalColor += col * d;
    }
    output.write(float4(finalColor, 1.0), gid);
}

// MARK: - Mandelbrot Generator
kernel void mandelbrotGeneratorKernel(texture2d<float, access::write> output [[texture(0)]], constant UniversalVisualizerParams& params [[buffer(0)]], uint2 gid [[thread_position_in_grid]]) {
    if (gid.x >= uint(params.outputSize.x) || gid.y >= uint(params.outputSize.y)) return;
    float2 res = params.outputSize; float T = params.time * params.param0; float2 uv = res; float2 u = float2(gid);
    uv = (u + u - float2(uv.x, 0.0)) / uv.y; float l = length(uv); float a = l * 0.5; float c = cos(a), s = sin(a);
    uv = float2x2(float2(c, s), float2(-s, c)) * uv; uv *= uv / l * 4.0;
    float3 color;
    for (int i = 0; i < 3; i++) {
        float v = length((uv - 0.5 * floor(uv / 0.5)) - uv.x * 0.25); float PI_local = 3.14159265;
        uv.y *= ((sin(v * 5.0)) * 0.5) * ((sin(T - length(uv) * 0.5) + PI_local) * 0.4);
        color[i] = cos(T + pow(length(uv), float(i) * 0.25));
    }
    float3 final_col = mix(color, float3(pow(uv.y, 1.0) * 0.5), length(uv) * 0.5);
    output.write(float4(final_col, 1.0), gid);
}

// MARK: - Apollonian Gasket Generator
static float apollonianGasket_df(float2 p, float iTime, float scale, float z) {
    float zoom = 0.5; p /= zoom; float tm = 0.1 * iTime; float c = cos(tm); float s = sin(tm); p = p * float2x2(c, s, -s, c);
    tm = 0.2 * iTime; float r = 0.5;
    float4 pp = float4(p.x, p.y, 0.0, 0.0) + float4(r*(0.5+0.5*sin(tm*sqrt(3.0))), r*(0.5+0.5*sin(tm*sqrt(1.5))), r*(0.5+0.5*sin(tm*sqrt(2.0))), 0.0);
    pp.w = 0.125 * (1.0 - tanh(length(pp.xyz))); float tmsqrt = tm * sqrt(0.5); float c2 = cos(tm); float s2 = sin(tm);
    float oldY = pp.y; pp.y = oldY * c2 + pp.z * s2; pp.z = -oldY * s2 + pp.z * c2;
    float c3 = cos(tmsqrt); float s3 = sin(tmsqrt); float oldX = pp.x; pp.x = oldX * c3 + pp.z * s3; pp.z = -oldX * s3 + pp.z * c3; pp /= z;
    for(int i=0; i<7; ++i) { pp = -1.0 + 2.0 * fract(0.5 * pp + 0.5); float r2 = dot(pp, pp); float k = 1.2 / r2; pp *= k; scale *= k; }
    return (abs(pp.y)/scale * z) * zoom;
}
kernel void apollonianGeneratorKernel(texture2d<float, access::write> output [[texture(0)]], constant UniversalVisualizerParams& params [[buffer(0)]], uint2 gid [[thread_position_in_grid]]) {
    if (gid.x >= uint(params.outputSize.x) || gid.y >= uint(params.outputSize.y)) return;
    float width = params.outputSize.x; float height = params.outputSize.y; float2 q = float2(gid) / float2(width, height); float2 p = -1.0 + 2.0 * q; p.x *= width / height;
    float aa = 2.0 / height; float lw = 0.0235; float lh = 1.25; float3 lp1 = float3(0.5, lh, 0.5); float3 lp2 = float3(-0.5, lh, 0.5);
    float z = params.param0 > 0 ? params.param0 : 1.0; float d = apollonianGasket_df(p, params.time, 1.0, z);
    float b = -0.125; float t = 10.0; float3 ro = float3(0.0, t, 0.0); float3 pp = float3(p.x, 0.0, p.y); float3 rd = normalize(pp - ro); float bt = -(t - b) / rd.y;
    float3 bp = ro + bt * rd; float3 srd1 = normalize(lp1 - bp); float3 srd2 = normalize(lp2 - bp); float bl21 = dot(lp1 - bp, lp1 - bp); float bl22 = dot(lp2 - bp, lp2 - bp);
    float st1 = (0.0 - b) / srd1.y; float3 sp1 = bp + srd1 * st1; float3 sp2 = bp + srd2 * st1;
    float sd1 = apollonianGasket_df(sp1.xz, params.time, 1.0, z); float sd2 = apollonianGasket_df(sp2.xz, params.time, 1.0, z);
    float3 col = float3(0.0); float ss = 15.0; col += float3(1.0) * (1.0 - exp(-ss * (max((sd1 + 0.0 * lw), 0.0)))) / bl21; col += float3(0.5) * (1.0 - exp(-ss * (max((sd2 + 0.0 * lw), 0.0)))) / bl22;
    float l = length(p); float hue = fract(0.75 * l - 0.3 * params.time) + 0.3 + 0.15; float sat = 0.75 * tanh(2.0 * l);
    float3 p3 = abs(fract(float3(hue) + float3(1.0, 2.0/3.0, 1.0/3.0)) * 6.0 - 3.0); float3 bcol = 1.0 * mix(float3(1.0), clamp(p3 - 1.0, 0.0, 1.0), sat);
    col *= (1.0 - tanh(0.75 * l)) * 0.5; col = mix(col, bcol, smoothstep(-aa, aa, -d)); col += 0.5 * sqrt(bcol.zxy) * (exp(-(10.0 + 100.0 * tanh(l)) * max(d, 0.0)));
    col = pow(clamp(col, 0.0, 1.0), float3(1.0/2.2)); col = col * params.param1 + 0.4 * col * col * (3.0 - 2.0 * col);
    col = mix(col, float3(dot(col, float3(0.333))), -0.4); col *= 0.5 + 0.5 * pow(19.0 * q.x * q.y * (1.0 - q.x) * (1.0 - q.y), 0.5);
    output.write(float4(col, 1.0), gid);
}

// MARK: - Rick And Morty Portal Generator
#define PORTAL_PI 3.14159265359 // define correctly
inline float rmportal_hash21(float2 co) { return fract(sin(dot(co, float2(12.9898, 78.233))) * 43758.5453); }
inline float2x2 rmportal_rmatrix(float a) { float c = cos(a); float s = sin(a); return float2x2(c, -s, s, c); }
inline float rmportal_S(float x) { return (3.0 * x * x - 2.0 * x * x * x); }
inline float rmportal_noise(float2 pos) {
    float2 i = floor(pos); float2 f = pos - i;
    float a = rmportal_hash21(i); float b = rmportal_hash21(i + float2(1.0, 0.0)); float c = rmportal_hash21(i + float2(0.0, 1.0)); float d = rmportal_hash21(i + float2(1.0, 1.0));
    float s1 = rmportal_S(f.x); float s2 = rmportal_S(f.y);
    return a + (b - a) * s1 + (c - a) * s2 + (a - b - c + d) * s1 * s2;
}
inline float rmportal_onoise(float2 pos) {
    float sum = 0.0; float power = 0.5; float delta = PORTAL_PI / 6.0;
    for (int i = 0; i < 3; i++) { sum += rmportal_noise(rmportal_rmatrix(delta * float(i)) * pos) * power; power *= 0.4; pos *= 1.9; } return sum;
}
inline float3 rmportal_portalTexture(float2 uv, float time) {
    float2 uv2; uv2.x = length(uv); uv2.y = (atan2(uv.y, uv.x) + PORTAL_PI) / (2.0 * PORTAL_PI);
    uv2.y = fract(uv2.y + uv2.x * 0.3 - time * 0.01); uv2.x = uv2.x + time * 0.3;
    float3 colors[4]; colors[0] = float3(0.184, 0.529, 0.086); colors[1] = float3(0.557, 0.890, 0.161); colors[2] = float3(0.349, 0.835, 0.110); colors[3] = float3(0.921, 0.980, 0.847);
    float br1 = rmportal_onoise(uv2 * 10.0); float br2 = rmportal_onoise(float2(uv2.x, uv2.y - 1.0) * 10.0); float br = mix(br1, br2, uv2.y);
    br = min(0.99, pow(br * 1.5, 2.5)); float scaledBr = clamp(br * 3.0, 0.0, 2.999); int idx = int(scaledBr); float frac_t = fract(scaledBr);
    return mix(colors[clamp(idx, 0, 3)], colors[clamp(idx + 1, 0, 3)], frac_t);
}
kernel void rickAndMortyPortalGeneratorKernel(texture2d<float, access::write> output [[texture(0)]], constant UniversalVisualizerParams& params [[buffer(0)]], uint2 gid [[thread_position_in_grid]]) {
    float width = params.outputSize.x; float height = params.outputSize.y;
    if (gid.x >= uint(width) || gid.y >= uint(height)) return;
    float2 uv = float2(gid) / params.outputSize; uv = (uv * 2.0 - 1.0) * float2(width / height, 1.0);
    output.write(float4(rmportal_portalTexture(uv, params.time), 1.0), gid);
}

// MARK: - Neon Rings Generator
static float4 nr_permute(float4 x) { float4 v = ((x * 34.0) + 1.0) * x; return v - 289.0 * floor(v / 289.0); }
static float4 nr_taylorInvSqrt(float4 r) { return 1.79284291400159 - 0.85373472095314 * r; }
static float nr_snoise(float3 v) {
    const float2 C = float2(1.0 / 6.0, 1.0 / 3.0); const float4 D = float4(0.0, 0.5, 1.0, 2.0);
    float3 i = floor(v + dot(v, C.yyy)); float3 x0 = v - i + dot(i, C.xxx);
    float3 g = step(x0.yzx, x0.xyz); float3 l = 1.0 - g; float3 i1 = min(g.xyz, l.zxy); float3 i2 = max(g.xyz, l.zxy);
    float3 x1 = x0 - i1 + C.xxx; float3 x2 = x0 - i2 + C.xxx * 2.0; float3 x3 = x0 - 1.0 + C.xxx * 3.0;
    i = i - 289.0 * floor(i / 289.0);
    float4 p = nr_permute(nr_permute(nr_permute(i.z + float4(0.0, i1.z, i2.z, 1.0)) + i.y + float4(0.0, i1.y, i2.y, 1.0)) + i.x + float4(0.0, i1.x, i2.x, 1.0));
    float n_ = 1.0 / 7.0; float3 ns = n_ * D.wyz - D.xzx; float4 j = p - 49.0 * floor(p * ns.z * ns.z);
    float4 x_ = floor(j * ns.z); float4 y_ = floor(j - 7.0 * x_); float4 x = x_ * ns.x + ns.yyyy; float4 y = y_ * ns.x + ns.yyyy;
    float4 h = 1.0 - abs(x) - abs(y); float4 b0 = float4(x.xy, y.xy); float4 b1 = float4(x.zw, y.zw);
    float4 s0 = floor(b0) * 2.0 + 1.0; float4 s1 = floor(b1) * 2.0 + 1.0; float4 sh = -step(h, float4(0.0));
    float4 a0 = b0.xzyw + s0.xzyw * sh.xxyy; float4 a1 = b1.xzyw + s1.xzyw * sh.zzww;
    float3 p0 = float3(a0.xy, h.x); float3 p1 = float3(a0.zw, h.y); float3 p2 = float3(a1.xy, h.z); float3 p3 = float3(a1.zw, h.w);
    float4 norm = nr_taylorInvSqrt(float4(dot(p0,p0), dot(p1,p1), dot(p2,p2), dot(p3,p3)));
    p0 *= norm.x; p1 *= norm.y; p2 *= norm.z; p3 *= norm.w;
    float4 m = max(0.6 - float4(dot(x0,x0), dot(x1,x1), dot(x2,x2), dot(x3,x3)), 0.0);
    m = m * m; return 42.0 * dot(m * m, float4(dot(p0,x0), dot(p1,x1), dot(p2,x2), dot(p3,x3)));
}
static float2 nr_hash23(float3 p3) {
    p3 = fract(p3 * float3(0.1031, 0.1030, 0.0973)); p3 += dot(p3, p3.yzx + 33.33); return fract((p3.xx + p3.yz) * p3.zy);
}
kernel void neonRingsGeneratorKernel(texture2d<float, access::write> output [[texture(0)]], constant UniversalVisualizerParams& params [[buffer(0)]], uint2 gid [[thread_position_in_grid]]) {
    float width = params.outputSize.x; float height = params.outputSize.y;
    if (gid.x >= uint(width) || gid.y >= uint(height)) return;
    float2 p = (float2(gid) - float2(width, height) * 0.5) / (height * 0.5);
    float time = params.time * params.param0; float ringCount = max(params.param1, 1.0); float thickness = params.param2;
    float beat = fract(time * 0.5); float back = max((beat - 7.0 / 8.0) / (1.0 / 8.0), 0.0);
    float timew = time - pow(0.5 - back, 2.0) * 1.0; float scale = 1.0 + pow(back, 2.0) * 4.0; float stepSz = mix(0.1, 0.15 - pow(fract(timew * 0.5), 1.2) * 0.05, 1.0);
    float3 col = float3(0.0); float len_p = length(p);
    for (float t = 0.0; t < 1.0; t += stepSz) {
        float to = stepSz * fract(timew * 0.5) + t; float tt = to * 32.0; float ringIdx = fract(tt / ringCount) * ringCount;
        float2 dir = (len_p > 0.001) ? normalize(p) : float2(1.0, 0.0); float n = nr_snoise(float3(dir * scale + float2(0.0, time * 0.5), time + to * 2.0));
        float offset = tt * 0.05; n *= 0.01 + tt * 0.01; float dist = abs(len_p - n - offset);
        float baseThickness = thickness * 1.8; float core = exp(-dist * 180.0 / baseThickness); float bloom = exp(-dist * 18.0 / baseThickness); float halo = exp(-dist * 5.0 / baseThickness) * 0.35;
        float glow = core * 1.2 + bloom * 0.7 + halo; glow *= smoothstep(1.0, 0.5, to);
        float hue = fract(ringIdx / ringCount + time * 0.1); float3 ringColor;
        ringColor.r = pow(max(0.0, cos(6.28318 * (hue + 0.00)) * 0.5 + 0.5), 0.5); ringColor.g = pow(max(0.0, cos(6.28318 * (hue + 0.33)) * 0.5 + 0.5), 0.5); ringColor.b = pow(max(0.0, cos(6.28318 * (hue + 0.67)) * 0.5 + 0.5), 0.5);
        float maxC = max(ringColor.r, max(ringColor.g, ringColor.b)); ringColor = mix(float3(maxC), ringColor, 2.0); ringColor = max(ringColor, float3(0.0));
        col += glow * ringColor;
    }
    col = 1.0 - exp(-col * 1.2); col = pow(col, float3(0.85, 0.9, 1.0)); col *= nr_hash23(float3(p * 120.0, time)).x * 0.12 + 0.88;
    output.write(float4(col, 1.0), gid);
}

// MARK: - Plasma Storm Generator
kernel void plasmaStormGeneratorKernel(texture2d<float, access::write> output [[texture(0)]], constant UniversalVisualizerParams& params [[buffer(0)]], uint2 gid [[thread_position_in_grid]]) {
    float width = params.outputSize.x; float height = params.outputSize.y;
    if (gid.x >= uint(width) || gid.y >= uint(height)) return;
    float2 uv = float2(gid) / float2(width, height); uv = (uv - 0.5) * 2.0; uv.x *= (width / height);
    float time = params.time * params.param0; float complexity = params.param1; float colorShift = params.param2; float saturation = params.param3; float bright = params.param4;
    float angle = time * 0.3; float ca = cos(angle); float sa = sin(angle); float2 ruv = float2(uv.x * ca - uv.y * sa, uv.x * sa + uv.y * ca);
    float v = 0.0; v += sin(ruv.x * 3.0 + time); v += sin(ruv.y * 3.0 + time * 0.7); v += sin((ruv.x + ruv.y) * 2.0 + time * 1.3); v += sin(length(ruv) * 4.0 - time * 0.9);
    if (complexity > 3.5) v += sin(ruv.x * 5.0 - ruv.y * 3.0 + time * 0.5); if (complexity > 4.5) v += sin(length(ruv - float2(sin(time), cos(time))) * 6.0);
    v /= max(complexity, 1.0); float3 col;
    col.r = sin(v * 3.14159 + colorShift * 6.28318 + 0.0) * 0.5 + 0.5; col.g = sin(v * 3.14159 + colorShift * 6.28318 + 2.094) * 0.5 + 0.5; col.b = sin(v * 3.14159 + colorShift * 6.28318 + 4.189) * 0.5 + 0.5;
    float gray = dot(col, float3(0.299, 0.587, 0.114)); col = mix(float3(gray), col, saturation); col *= bright;
    output.write(float4(col, 1.0), gid);
}

// MARK: - Generic Utils
static float glslmod(float a, float b) { return a - b * floor(a / b); }
static float2 glslmod2f(float2 a, float b) { return a - b * floor(a / b); }
static float3 glslmod3f(float3 a, float b) { return a - b * floor(a / b); }
static float4 glslmod4f(float4 a, float b) { return a - b * floor(a / b); }

// MARK: - Cosmic Smoke
static float cs_hash21(float2 p) { float3 p3 = fract(float3(p.xyx) * 0.1031); p3 += dot(p3, p3.yzx + 33.33); return fract((p3.x + p3.y) * p3.z); }
static float cs_noise(float2 p) { float2 i = floor(p); float2 f = fract(p); float2 u = f * f * (3.0 - 2.0 * f); float a = cs_hash21(i); float b = cs_hash21(i + float2(1.0, 0.0)); float c = cs_hash21(i + float2(0.0, 1.0)); float d = cs_hash21(i + float2(1.0, 1.0)); return mix(mix(a, b, u.x), mix(c, d, u.x), u.y); }
static float cs_fbm(float2 p) { float val = 0.0; float amp = 0.5; for (int i = 0; i < 6; i++) { val += amp * cs_noise(p); p *= 2.0; amp *= 0.5; } return val; }
static float cs_dualfbm(float2 p, float t) { float2 q = float2(cs_fbm(p + float2(0.0, 0.0)), cs_fbm(p + float2(5.2, 1.3))); float2 r = float2(cs_fbm(p + 4.0 * q + float2(1.7, 9.2) + 0.15 * t), cs_fbm(p + 4.0 * q + float2(8.3, 2.8) + 0.126 * t)); return cs_fbm(p + 4.0 * r); }
static float cs_circ(float2 p) { float r = length(p); return log(abs(r)); }

kernel void cosmicSmokeKernel(texture2d<float, access::write> output [[texture(0)]], constant UniversalVisualizerParams& params [[buffer(0)]], uint2 gid [[thread_position_in_grid]]) {
    if (gid.x >= uint(params.outputSize.x) || gid.y >= uint(params.outputSize.y)) return;
    float2 res = params.outputSize; float2 fragCoord = float2(gid); float speed = params.param0; float t = params.time * speed;
    float2 p = fragCoord / res.y - float2(res.x / res.y * 0.5, 0.5); p *= 4.0;
    float angle = t * 0.2; float ca = cos(angle), sa = sin(angle); p = float2(p.x * ca - p.y * sa, p.x * sa + p.y * ca);
    float rz = cs_dualfbm(p, t); float ringPhase = fract(t * 0.15); p /= (0.5 + ringPhase * 2.0); rz *= pow(abs(0.1 - cs_circ(p)), 0.9);
    float3 col = float3(0.2, 0.1, 0.4) / rz; col = pow(abs(col), float3(0.99));
    float2 vig = (fragCoord / res - 0.5) * 2.0; col *= 1.0 - dot(vig, vig) * 0.3;
    output.write(float4(col, 1.0), gid);
}

// MARK: - Menger Tunnel
static float2x2 mt_rot(float a) { float c = cos(a), s = sin(a); return float2x2(float2(c, s), float2(-s, c)); }
static float2 mt_pmod(float2 p, float r, float t) {
    float PI_local = 3.14159265; float PI2 = 6.28318530; float a = atan2(p.x, p.y) + PI_local / r; float sinVal = sin(0.42 * t);
    float n = PI2 * (-sinVal * sinVal + 0.35) / r; a = floor(a / n) * n - PI_local; p = mt_rot(-a) * p; return p;
}
static float mt_crossf(float3 r, float w) { float da = max(r.x, r.y); float db = max(r.y, r.z); float dc = max(r.z, r.x); return min(da, min(db, dc)) - w; }
static float mt_boxf(float3 p, float3 b) { float3 d = abs(p) - b; return min(max(d.x, max(d.y, d.z)), 0.0) + length(max(d, 0.0)); }
static float mt_mengersponge(float3 p, float scale, float width) {
    float d = mt_boxf(p, float3(1.0)); float s = 1.0;
    for (int m = 0; m < 5; m++) { float3 a = p * s - 2.0 * floor(p * s / 2.0) - 1.0; s *= scale; float3 r = abs(1.0 - scale * abs(a)); float c = mt_crossf(r, width) / s; d = max(d, c); } return d;
}
static float mt_map(float3 p) { p = glslmod3f(p, 4.0) - 2.0; return mt_mengersponge(p, 3.0, 1.0); }
kernel void mengerTunnelKernel(texture2d<float, access::write> output [[texture(0)]], constant UniversalVisualizerParams& params [[buffer(0)]], uint2 gid [[thread_position_in_grid]]) {
    if (gid.x >= uint(params.outputSize.x) || gid.y >= uint(params.outputSize.y)) return;
    float2 res = params.outputSize; float speed = params.param0; float t = params.time * speed; float2 uv = (float2(gid) * 2.0 - res) / min(res.x, res.y);
    float3 ro = float3(0.0, 0.0, 8.0 * t); float3 rd = normalize(float3(uv, 0.8 * sin(0.58 * t)));
    float dp = 0.0; float d = 0.0; float3 col = float3(0.0);
    for (int i = 0; i < 99; i++) {
        float3 pos = ro + rd * dp; pos.xy = mt_rot(0.1 * t) * pos.xy; pos.xy = mt_pmod(pos.xy, 8.0, t); d = mt_map(pos);
        if (d < 0.00001) { col = float3(1.0 - float(i) * 0.02); break; } dp += d * 0.8;
    }
    output.write(float4(col, 1.0), gid);
}

// MARK: - Sponge Tunnel
static float2 st_pmod(float2 p, float r, float t) {
    float PI_local = 3.14159265; float PI2 = 6.28318530; float a = atan2(p.x, p.y) + PI_local / r; float sinVal = sin(0.42 * t);
    float n = PI2 * (-sinVal * sinVal + 0.35) / r; a = floor(a / n) * n - PI_local; p = mt_rot(a) * p; return p;
}
kernel void spongeTunnelKernel(texture2d<float, access::write> output [[texture(0)]], constant UniversalVisualizerParams& params [[buffer(0)]], uint2 gid [[thread_position_in_grid]]) {
    if (gid.x >= uint(params.outputSize.x) || gid.y >= uint(params.outputSize.y)) return;
    float2 res = params.outputSize; float speed = params.param0; float t = params.time * speed; float2 uv = (float2(gid) * 2.0 - res) / min(res.x, res.y);
    float3 ro = float3(0.0, 0.0, 8.0 * t); float3 rd = normalize(float3(uv, 0.8 * sin(0.58 * t)));
    float dp = 0.0; float d = 0.0; float3 col = float3(0.0);
    for (int i = 0; i < 99; i++) {
        float3 pos = ro + rd * dp; pos.xy = mt_rot(-0.1 * t) * pos.xy; pos.xy = st_pmod(pos.xy, 8.0, t); d = mt_map(pos);
        if (d < 0.00001) { col = float3(1.0 - float(i) * 0.02); break; } dp += d * 0.8;
    }
    output.write(float4(col, 1.0), gid);
}

// MARK: - Kaleidoscope Tunnel
static float2x2 tg_rot(float a) { float c = cos(a), s = sin(a); return float2x2(float2(c, s), float2(-s, c)); }
static void tg_foldRotate(thread float2& p, float s) {
    float PI_local = 3.14159265; float a = PI_local / s - atan2(p.x, p.y); float n = 2.0 * PI_local / s; a = floor(a / n) * n;
    float c = cos(a), s2 = sin(a); p = float2x2(float2(c, s2), float2(-s2, c)) * p;
}
static float tg_sdRect(float2 p, float2 s) { float2 d = abs(p) - s; return min(max(d.x, d.y), 0.0) + length(max(d, 0.0)); }
static float tg_tex(float2 p, float z) {
    float PI_local = 3.14159265; tg_foldRotate(p, 8.0); float2 q = (fract(p / 10.0) - 0.5) * 10.0;
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 2; j++) { q = abs(q) - 0.25; q = tg_rot(PI_local * 0.25) * q; }
        q = abs(q) - float2(1.0, 1.5); q = tg_rot(PI_local * 0.25 * z) * q; tg_foldRotate(q, 3.0);
    }
    return smoothstep(0.9, 1.0, 1.0 / (1.0 + abs(tg_sdRect(q, float2(1.0)))));
}
static float tg_Bokeh(float2 p, float2 sp, float size, float mi, float blur) { float d = length(p - sp); return smoothstep(size, size * (1.0 - blur), d) * mix(mi, 1.0, smoothstep(size * 0.8, size, d)); }
static float2 tg_hash(float2 p) { p = float2(dot(p, float2(127.1, 311.7)), dot(p, float2(269.5, 183.3))); return fract(sin(p) * 43758.5453) * 2.0 - 1.0; }
static float tg_dirt(float2 uv, float n) { float2 p = fract(uv * n); float2 st = (floor(uv * n) + 0.5) / n; float2 rnd = tg_hash(st); float2 center = float2(0.5) + 0.2 * rnd; return tg_Bokeh(p, center, 0.05, abs(rnd.y * 0.4) + 0.3, 0.25 + rnd.x * rnd.y * 0.2); }
static float tg_sm(float start, float end, float t, float smo) { return smoothstep(start, start + smo, t) - smoothstep(end - smo, end, t); }
kernel void kaleidoscopeTunnelKernel(texture2d<float, access::write> output [[texture(0)]], constant UniversalVisualizerParams& params [[buffer(0)]], uint2 gid [[thread_position_in_grid]]) {
    if (gid.x >= uint(params.outputSize.x) || gid.y >= uint(params.outputSize.y)) return;
    float2 res = params.outputSize; float speed = params.param0; float time = params.time * speed;
    float2 uv = float2(gid) / res; uv = (uv * 2.0 - 1.0); uv.x *= res.x / res.y; uv *= 2.0; float3 col = float3(0.0);
    int NN = 6; float NNf = float(NN); float INTERVAL = 3.0;
    for (int ii = 0; ii < NN; ii++) {
        float iif = float(NN - ii);
        float t_sub1 = iif * INTERVAL - glslmod(time - INTERVAL * 0.75, INTERVAL); float3 INT1 = float3((NNf * INTERVAL - t_sub1) / (NNf * INTERVAL));
        float2 p1 = glslmod2f(uv * max(0.0, t_sub1) * 0.1 + float2(0.2, -0.2) * time, 1.2); col = mix(col, INT1, tg_dirt(p1, 3.5));
        float t_sub2 = iif * INTERVAL - glslmod(time + INTERVAL * 0.5, INTERVAL); float3 INT2 = float3((NNf * INTERVAL - t_sub2) / (NNf * INTERVAL));
        float tx2 = tg_tex(uv * max(0.0, t_sub2), 4.45); col = mix(col, INT2 * float3(0.7, 0.8, 1.0) * 1.3, tx2);
        float t_sub3 = iif * INTERVAL - glslmod(time - INTERVAL * 0.25, INTERVAL); float3 INT3 = float3((NNf * INTERVAL - t_sub3) / (NNf * INTERVAL));
        float2 p3 = glslmod2f(uv * max(0.0, t_sub3) * 0.1 + float2(-0.2, -0.2) * time, 1.2); col = mix(col, INT3 * 1.0, tg_dirt(p3, 3.5));
        float t_sub4 = iif * INTERVAL - glslmod(time, INTERVAL); float3 INT4 = float3((NNf * INTERVAL - t_sub4) / (NNf * INTERVAL));
        float r = length(uv * 2.0 * max(0.0, t_sub4)); float rr = tg_sm(-24.0, 0.0, r - glslmod(time * 30.0, 90.0), 10.0);
        float tx4 = tg_tex(uv * 2.0 * max(0.0, t_sub4), 0.27 + 2.0 * rr); col = mix(col, mix(INT4 * 1.0, INT4 * float3(0.7, 0.5, 1.0) * 3.0, rr), tx4);
    }
    output.write(float4(clamp(col, 0.0, 1.0), 1.0), gid);
}

// MARK: - Warp Bump
static float2 wb_W(float2 p, float t) {
    p = (p + 3.0) * 4.0; t = t / 2.0;
    for (int i = 0; i < 3; i++) {
        p += cos(float2(p.y, p.x) * 3.0 + float2(t, 1.57)) / 3.0; p += sin(float2(p.y, p.x) + t + float2(1.57, 0.0)) / 2.0; p *= 1.3;
    }
    p += fract(sin(p + float2(13.0, 7.0)) * 5e5) * 0.03 - 0.015; return glslmod2f(p, 2.0) - 1.0;
}
static float wb_bumpFunc(float2 p, float t) { return length(wb_W(p, t)) * 0.7071; }
static float3 wb_smoothFract(float3 x) { x = fract(x); return min(x, x * (1.0 - x) * 12.0); }
kernel void warpBumpKernel(texture2d<float, access::write> output [[texture(0)]], constant UniversalVisualizerParams& params [[buffer(0)]], uint2 gid [[thread_position_in_grid]]) {
    if (gid.x >= uint(params.outputSize.x) || gid.y >= uint(params.outputSize.y)) return;
    float2 res = params.outputSize; float t = params.time * params.param0; float bumpFactor = params.param1;
    float2 uv = (float2(gid) - res * 0.5) / res.y; float3 sp = float3(uv, 0.0); float3 rd = normalize(float3(uv, 1.0));
    float3 lp = float3(cos(t) * 0.5, sin(t) * 0.2, -1.0); float3 sn = float3(0.0, 0.0, -1.0);
    float eps = 4.0 / res.y; float f = wb_bumpFunc(sp.xy, t);
    float fx = (wb_bumpFunc(sp.xy - float2(eps, 0.0), t) - f) / eps; float fy = (wb_bumpFunc(sp.xy - float2(0.0, eps), t) - f) / eps;
    sn = normalize(sn + float3(fx, fy, 0.0) * bumpFactor);
    float3 ld = normalize(lp - sp); float diff = max(dot(sn, ld), 0.0); diff = pow(diff, 4.0) * 0.66 + pow(diff, 8.0) * 0.34;
    float spec = pow(max(dot(reflect(-ld, sn), -rd), 0.0), 12.0); float atten = 1.0 / (1.0 + length(lp - sp) * length(lp - sp) * 0.15); atten *= f * 0.9 + 0.1;
    float2 wVal = wb_W(sp.xy, t); float3 texCol = wb_smoothFract(float3(wVal.x, wVal.y, wVal.y)) * 0.1 + 0.2;
    float3 col = (texCol * (diff * float3(1.0, 0.97, 0.92) * 2.0 + 0.5) + float3(1.0, 0.6, 0.2) * spec * 2.0) * atten;
    float envRef = max(dot(reflect(rd, sn), float3(1.0, 0.0, 0.0)), 0.0); col += col * pow(envRef, 4.0) * float3(0.25, 0.5, 1.0) * 3.0;
    output.write(float4(sqrt(clamp(col, 0.0, 1.0)), 1.0), gid);
}

// MARK: - Cave Tunnel
static float ct_hash12(float2 p) { float h = dot(p, float2(127.1, 311.7)); return fract(sin(h) * 43758.5453123); }
static float ct_noise3(float3 p) {
    float3 i = floor(p); float3 f = fract(p); float3 fm = f - 1.0; float3 u = 1.0 + fm * fm * fm * fm * fm;
    float2 ii = i.xy + i.z * float2(5.0);
    float v1 = mix(mix(ct_hash12(ii), ct_hash12(ii + float2(1.0, 0.0)), u.x), mix(ct_hash12(ii + float2(0.0, 1.0)), ct_hash12(ii + float2(1.0, 1.0)), u.x), u.y);
    ii += float2(5.0);
    float v2 = mix(mix(ct_hash12(ii), ct_hash12(ii + float2(1.0, 0.0)), u.x), mix(ct_hash12(ii + float2(0.0, 1.0)), ct_hash12(ii + float2(1.0, 1.0)), u.x), u.y);
    return max(mix(v1, v2, u.z), 0.0);
}
static float ct_fbm(float3 p) { float r = 0.0; float w = 1.0, s = 1.0; for (int i = 0; i < 4; i++) { w *= 0.25; s *= 3.0; r += w * ct_noise3(s * p); } return r; }
static float ct_yC(float x) { return cos(x * -0.134) * 1.0 * sin(x * 0.13) * 15.0 + ct_fbm(float3(x * 0.1, 0.0, 0.0) * 55.4); }
static float ct_fCylinderInf(float3 p, float r) { return length(p.xz) - r; }
static float ct_mapScene(float3 p, float t) {
    p.x -= ct_yC(p.y * 0.1) * 3.0; p.z += ct_yC(p.y * 0.01) * 4.0;
    float n = pow(abs(ct_fbm(p * 0.06)) * 12.0, 1.3); float s = ct_fbm(p * 0.01 + float3(0.0, t * 0.14, 0.0)) * 128.0;
    float dist = max(0.0, -ct_fCylinderInf(p, s + 18.0 - n));
    float3 p2 = p; p2.x -= sin(p.y * 0.02) * 34.0 + cos(p.z * 0.01) * 62.0; dist = max(dist, -ct_fCylinderInf(p2, s + 28.0 + n * 2.0)); return dist;
}
kernel void caveTunnelKernel(texture2d<float, access::write> output [[texture(0)]], constant UniversalVisualizerParams& params [[buffer(0)]], uint2 gid [[thread_position_in_grid]]) {
    if (gid.x >= uint(params.outputSize.x) || gid.y >= uint(params.outputSize.y)) return;
    float2 res = params.outputSize; float T = params.time * params.param0; float2 uv = (float2(gid) / res - 0.5) * 2.8 * float2(res.x / res.y, 1.0);
    float3 ro = float3(0.0, 30.0 + T * 100.0, -0.1); ro.x += ct_yC(ro.y * 0.1) * 3.0; ro.z -= ct_yC(ro.y * 0.01) * 4.0;
    float3 vrp = float3(0.0, 50.0 + T * 100.0, 2.0); vrp.x += ct_yC(vrp.y * 0.1) * 3.0; vrp.z -= ct_yC(vrp.y * 0.01) * 4.0;
    float3 vuv = normalize(float3(cos(T), sin(T * 0.11), sin(T * 0.41))); float3 vpn = normalize(vrp - ro);
    float3 u = normalize(cross(vuv, vpn)); float3 v = cross(vpn, u); float3 rd = normalize(ro + vpn + uv.x * u + uv.y * v - ro);
    float tt = 10.0; float candidateT = 10.0; float candidateErr = 1e32; float prevRadius = 0.0; float stepLength = 0.0;
    float functionSign = ct_mapScene(ro, T) < 0.0 ? -1.0 : 1.0; float omega = 1.3; int finalIter = 0;
    for (int i = 0; i < 100; i++) {
        float3 pos = rd * tt + ro; float mp = ct_mapScene(pos, T); finalIter = i;
        float signedRadius = functionSign * mp; float radius = abs(signedRadius); bool sorFail = omega > 1.0 && (radius + prevRadius) < stepLength;
        if (sorFail) { stepLength -= omega * stepLength; omega = 1.0; } else { stepLength = signedRadius * omega; }
        prevRadius = radius; float error = radius / tt;
        if (!sorFail && error < candidateErr) { candidateT = tt; candidateErr = error; }
        if ((!sorFail && error < 0.001) || tt > 1e3) break;
        tt += stepLength * 0.5;
    }
    float trDist = (tt > 1e3 || candidateErr > 0.001) ? 1e32 : candidateT; float3 hit = ro + rd * trDist;
    float3 col = float3(1.0, 0.5, 0.4) * ct_fbm(hit.xzy * 0.01) * 20.0; col.b *= ct_fbm(hit * 0.01) * 10.0;
    col = min(0.8, float(finalIter) / 90.0) * col + col * 0.03; col *= 1.0 + 0.9 * (abs(ct_fbm(hit * 0.002 + 3.0) * 10.0) * (ct_fbm(float3(0.0, 0.0, T * 0.05) * 2.0))); col *= 0.6;
    float distC = trDist; float fogF = 0.0;
    for (float fi = 0.0; fi < 24.0; fi++) {
        fogF += ct_fbm((hit - rd * distC) * float3(0.1, 0.1, 0.1) * 0.3) * 0.1; distC -= 3.0; if (distC < 3.0) break;
    }
    col += float3(0.0, 0.4, 0.5) * pow(abs(fogF * 1.5), 3.0) * 4.0;
    float3 colFinal = clamp(col * (1.0 - length(uv) / 2.0), 0.0, 1.0);
    output.write(float4(pow(abs(colFinal / max(trDist, 0.001) * 130.0), float3(0.8)), 1.0), gid);
}

// MARK: - Monster
static float2 mn_rot(float2 p, float r) { float c = cos(r), s = sin(r); return float2(p.x * c - p.y * s, p.x * s + p.y * c); }
static float2 mn_pmod(float2 p, float n) { float TAU = 6.28318; float np = TAU / n; float r = atan2(p.x, p.y) - 0.5 * np; r = glslmod(r, np) - 0.5 * np; return length(p) * float2(cos(r), sin(r)); }
static float mn_cube(float3 p, float3 s) { float3 q = abs(p); float3 m = max(s - q, 0.0); return length(max(q - s, 0.0)) - min(m.x, min(m.y, m.z)); }
static float mn_dist(float3 p, float t) {
    p.z -= t; p.xy = mn_rot(p.xy, p.z); p.xy = mn_pmod(p.xy, 6.0); float k = 0.7; float zid = floor(p.z * k); p = glslmod3f(p, k) - 0.5 * k;
    for (int i = 0; i < 4; i++) { p = abs(p) - 0.3; p.xy = mn_rot(p.xy, 1.0 + zid + 0.1 * t); p.xz = mn_rot(p.xz, 1.0 + 4.7 * zid + 0.3 * t); }
    return min(mn_cube(p, float3(0.3)), length(p) - 0.4);
}
kernel void monsterKernel(texture2d<float, access::write> output [[texture(0)]], constant UniversalVisualizerParams& params [[buffer(0)]], uint2 gid [[thread_position_in_grid]]) {
    if (gid.x >= uint(params.outputSize.x) || gid.y >= uint(params.outputSize.y)) return;
    float2 res = params.outputSize; float t = params.time * params.param0; float2 uv = float2(gid) / res; uv = 2.0 * (uv - 0.5); uv.y *= res.y / res.x; uv = mn_rot(uv, t);
    float3 ro = float3(0.0, 0.0, 0.1); float3 rd = normalize(float3(uv, 0.0) - ro); float totalD = 2.0; float ac = 0.0; float3 pn = float3(0.0);
    for (int i = 0; i < 66; i++) {
        pn = ro + rd * totalD; float d = max(0.0, abs(mn_dist(pn, t) * 0.2));
        if (d < 0.001) ac += 0.1; totalD += max(d, 0.001);
    }
    float3 col = float3(0.1, 0.7, 0.7) * 0.2 * ac; float3 ep = pn; ep.z += -1.5 * t;
    float em = clamp(0.01 / max(abs(glslmod(ep.z, 0.5) - 0.25), 0.0001), 0.0, 100.0); col += 3.0 * em * float3(0.1, 1.0, 0.1);
    output.write(float4(clamp(col, 0.0, 1.0), 1.0), gid);
}

// MARK: - Phantom Star
static float2x2 ps_rot(float a) { float c = cos(a), s = sin(a); return float2x2(float2(c, s), float2(-s, c)); }
static float2 ps_pmod(float2 p, float r) {
    float a = atan2(p.x, p.y) + 3.14159265 / r; float n = 6.28318530 / r; a = floor(a / n) * n; p = ps_rot(-a) * p; return p;
}
static float ps_box(float3 p, float3 b) { float3 d = abs(p) - b; return min(max(d.x, max(d.y, d.z)), 0.0) + length(max(d, 0.0)); }
static float ps_ifsBox(float3 p, float t) {
    for (int i = 0; i < 5; i++) { p = abs(p) - 1.0; p.xy = ps_rot(t * 0.3) * p.xy; p.xz = ps_rot(t * 0.1) * p.xz; }
    p.xz = ps_rot(t) * p.xz; return ps_box(p, float3(0.4, 0.8, 0.3));
}
static float ps_map(float3 p, float t) {
    p.x = glslmod(p.x - 5.0, 10.0) - 5.0; p.y = glslmod(p.y - 5.0, 10.0) - 5.0; p.z = glslmod(p.z, 16.0) - 8.0;
    p.xy = ps_pmod(p.xy, 5.0); return ps_ifsBox(p, t);
}
kernel void phantomStarKernel(texture2d<float, access::write> output [[texture(0)]], constant UniversalVisualizerParams& params [[buffer(0)]], uint2 gid [[thread_position_in_grid]]) {
    if (gid.x >= uint(params.outputSize.x) || gid.y >= uint(params.outputSize.y)) return;
    float2 res = params.outputSize; float t = params.time * params.param0; float2 p = (float2(gid) * 2.0 - res) / min(res.x, res.y);
    float3 cPos = float3(0.0, 0.0, -3.0 * t); float3 cDir = normalize(float3(0.0, 0.0, -1.0)); float3 cUp = float3(sin(t), 1.0, 0.0);
    float3 ray = normalize(cross(cDir, cUp) * p.x + cUp * p.y + cDir);
    float acc = 0.0, acc2 = 0.0, td = 0.0;
    for (int i = 0; i < 99; i++) {
        float3 pos = cPos + ray * td; float dist = max(abs(ps_map(pos, t)), 0.02); float a = exp(-dist * 3.0);
        if (glslmod(length(pos) + 24.0 * t, 30.0) < 3.0) { a *= 2.0; acc2 += a; }
        acc += a; td += dist * 0.5;
    }
    output.write(float4(float3(acc * 0.01, acc * 0.011 + acc2 * 0.002, acc * 0.012 + acc2 * 0.005), 1.0), gid);
}

// MARK: - Glowing Marble
kernel void glowingMarbleKernel(texture2d<float, access::write> output [[texture(0)]], constant UniversalVisualizerParams& params [[buffer(0)]], uint2 gid [[thread_position_in_grid]]) {
    if (gid.x >= uint(params.outputSize.x) || gid.y >= uint(params.outputSize.y)) return;
    float2 res = params.outputSize; float t = params.time * params.param0; float2 uv = (2.0 * float2(gid) - res) / min(res.x, res.y);
    for (float i = 1.0; i < 10.0; i++) { uv.x += 0.6 / i * cos(i * 2.5 * uv.y + t); uv.y += 0.6 / i * cos(i * 1.5 * uv.x + t); }
    output.write(float4(float3(0.1) / abs(sin(t - uv.y - uv.x)), 1.0), gid);
}

// MARK: - Hex Truchet
static float2 ht_hash22(float2 p) { float n = sin(glslmod(dot(p, float2(41, 289)), 6.2831589)); return fract(float2(262144, 32768) * n) * 0.75 + 0.25; }
static float ht_n3D(float3 p) {
    const float3 s = float3(7, 157, 113); float3 ip = floor(p); p -= ip; float4 h = float4(0.0, s.yz, s.y + s.z) + dot(ip, s); p = p * p * (3.0 - 2.0 * p);
    h = mix(fract(sin(glslmod4f(h, 6.2831589)) * 43758.5453), fract(sin(glslmod4f(h + s.x, 6.2831589)) * 43758.5453), p.x); h.xy = mix(h.xz, h.yw, p.y); return mix(h.x, h.y, p.z);
}
static float3 ht_envMap(float3 rd, float3 sn, float t) {
    float3 sRd = rd; rd.xy -= t * 0.25; rd *= 3.0; float c = smoothstep(0.4, 1.0, ht_n3D(rd) * 0.57 + ht_n3D(rd * 2.0) * 0.28 + ht_n3D(rd * 4.0) * 0.15);
    float3 col = float3(c, c * c, c * c * c * c); return mix(col, col.yzx, sRd * 0.25 + 0.25);
}
static float ht_heightMap(float2 p) {
    p *= 3.0; float2 h = float2(p.x + p.y * 0.57735, p.y * 1.1547); float2 fh = floor(h); float2 f = h - fh; h = fh;
    float c = fract((h.x + h.y) / 3.0); h = c < 0.666 ? (c < 0.333 ? h : h + 1.0) : h + step(f.yx, f);
    p -= float2(h.x - h.y * 0.5, h.y * 0.8660254); c = fract(cos(dot(h, float2(41, 289))) * 43758.5453); p -= p * step(c, 0.5) * 2.0; p -= float2(-1, 0); c = dot(p, p);
    p -= float2(1.5, 0.8660254); c = min(c, dot(p, p)); p -= float2(0, -1.73205); c = min(c, dot(p, p)); return sqrt(c);
}
static float ht_map(float3 p) { float c = clamp((cos(ht_heightMap(p.xy) * 6.2831589) + cos(ht_heightMap(p.xy) * 6.2831589 * 2.0)) * 0.6 + 0.5, 0.0, 1.0); return 1.0 - p.z - c * 0.025; }
static float3 ht_getNormal(float3 p, thread float &edge, thread float &crv) {
    float2 e = float2(0.01, 0.0); float d1 = ht_map(p + e.xyy), d2 = ht_map(p - e.xyy), d3 = ht_map(p + e.yxy), d4 = ht_map(p - e.yxy), d5 = ht_map(p + e.yyx), d6 = ht_map(p - e.yyx), d = ht_map(p) * 2.0;
    edge = smoothstep(0.0, 1.0, sqrt((abs(d1 + d2 - d) + abs(d3 + d4 - d) + abs(d5 + d6 - d)) / e.x * 2.0)); crv = clamp((d1 + d2 + d3 + d4 + d5 + d6 - d * 3.0) * 32.0 + 0.6, 0.0, 1.0);
    e = float2(0.0025, 0.0); return normalize(float3(ht_map(p + e.xyy) - ht_map(p - e.xyy), ht_map(p + e.yxy) - ht_map(p - e.yxy), ht_map(p + e.yyx) - ht_map(p - e.yyx)));
}
static float ht_calculateAO(float3 p, float3 n) { float sca = 2.0, occ = 0.0; for (float i = 0.0; i < 5.0; i++) { float hr = 0.01 + i * 0.5 / 4.0; occ += (hr - ht_map(n * hr + p)) * sca; sca *= 0.7; } return clamp(1.0 - occ, 0.0, 1.0); }
static float ht_voronoi(float2 p) {
    float2 g = floor(p); p -= g; float3 d = float3(1.0);
    for (int y = -1; y <= 1; y++) { for (int x = -1; x <= 1; x++) { float2 o = float2(float(x), float(y)); o += ht_hash22(g + o) - p; d.z = dot(o, o); d.y = max(d.x, min(d.y, d.z)); d.x = min(d.x, d.z); } }
    return max(d.y / 1.2 - d.x * 1.0, 0.0) / 1.2;
}
kernel void hexTruchetKernel(texture2d<float, access::write> output [[texture(0)]], constant UniversalVisualizerParams& params [[buffer(0)]], uint2 gid [[thread_position_in_grid]]) {
    if (gid.x >= uint(params.outputSize.x) || gid.y >= uint(params.outputSize.y)) return;
    float2 res = params.outputSize; float t = params.time * params.param0; float tm = t / 2.0;
    float3 rd = normalize(float3(2.0 * float2(gid) - res, res.y)); float2 a = sin(float2(1.570796, 0.0) + sin(tm / 4.0) * 0.3); rd.xy = float2x2(float2(a.x, a.y), float2(-a.y, a.x)) * rd.xy;
    float3 ro = float3(tm, cos(tm / 4.0), 0.0); float3 lp = ro + float3(cos(tm / 2.0) * 0.5, sin(tm / 2.0) * 0.5, -0.5);
    float d, tt = 0.0; for (int j = 0; j < 32; j++) { d = ht_map(ro + rd * tt); tt += d * 0.7; if (d < 0.001) break; }
    float edge, crv; float3 sp = ro + rd * tt; float3 sn = ht_getNormal(sp, edge, crv); float3 ld = normalize(lp - sp);
    float3 fold = cos(float3(1, 2, 4) * ht_heightMap(sp.xy) * 6.2831589); float c2 = clamp(cos(ht_heightMap((sp.xy + sp.z * 0.025) * 6.0) * 6.2831589 * 3.0) + 0.5, 0.0, 1.0);
    float3 oC = float3(1.0); if (fold.x > 0.0) oC = float3(1, 0.05, 0.1) * c2; if (fold.x < 0.05 && fold.y < 0.0) oC = float3(1, 0.7, 0.45) * (c2 * 0.25 + 0.75); else if (fold.x < 0.0) oC = float3(1, 0.8, 0.4) * c2;
    float p1 = (1.0 - smoothstep(0.0, 0.1, ht_voronoi(sp.xy * 4.0 + float2(tm, cos(tm / 4.0)))) + 0.25) * (1.0 - smoothstep(0.0, 0.1, fold.x * 0.5 + 0.5)); oC += oC.yxz * p1 * p1;
    float3 col = oC * (max(dot(ld, sn), 0.0) + 0.5) + float3(1.0, 0.7, 0.4) * pow(max(dot(reflect(-ld, sn), -rd), 0.0), 16.0) * 2.0 + float3(0.4, 0.7, 1.0) * pow(clamp(dot(sn, rd) + 1.0, 0.0, 1.0), 3.0);
    col += (oC * 0.5 + 0.5) * ht_envMap(reflect(rd, sn), sn, tm) * 6.0;
    output.write(float4(sqrt(clamp(col * (1.0 - edge * 0.85) * (1.0 / (1.0 + max(length(lp - sp), 0.001) * 0.125)) * (crv * 0.9 + 0.1) * ht_calculateAO(sp, sn), 0.0, 1.0)), 1.0), gid);
}

// MARK: - Matrix Rain
static float mr_rand(float2 co) { return fract(sin(dot(co, float2(12.9898, 78.233))) * 43758.5453); }
static float mr_rchar(float2 outer, float2 inner, float globalTime) {
    float2 seed = floor(inner * 4.0) + outer.y; if (mr_rand(float2(outer.y, 23.0)) > 0.98) seed += floor((globalTime + mr_rand(float2(outer.y, 49.0))) * 3.0);
    return float(mr_rand(seed) > 0.5);
}
static float4 mr_layer(float2 fragCoord, float2 iResolution, float globalTime, float xOffset) {
    float2 position = fragCoord / iResolution; position.x += xOffset; position.x /= iResolution.x / iResolution.y;
    float rx = fragCoord.x / (40.0 * 3.0); float mx = 40.0 * 3.0 * fract(position.x * 30.0 * 3.0);
    if (mx > 12.0 * 3.0) return float4(0.0);
    float ry = position.y * (xOffset == 0.0 ? 600.0 : 700.0) + mr_rand(float2(floor(rx), floor(rx) * 3.0)) * 100000.0 + globalTime * mr_rand(float2(floor(fragCoord.x / (xOffset == 0.0 ? 15.0 : 12.0)), 23.0)) * 120.0;
    float my = glslmod(ry, 15.0); if (my > 12.0 * 3.0) return float4(0.0);
    float col_v = max(glslmod(-floor(ry / 15.0), 24.0) - 4.0, 0.0) / 20.0;
    return float4((col_v < 0.8 ? float3(0.0, col_v / 0.8, 0.0) : mix(float3(0.0, 1.0, 0.0), float3(1.0), (col_v - 0.8) / 0.2)) * mr_rchar(float2(rx, floor(ry / 15.0)), float2(mx, my) / 12.0, globalTime), 1.0);
}
kernel void matrixRainKernel(texture2d<float, access::write> output [[texture(0)]], constant UniversalVisualizerParams& params [[buffer(0)]], uint2 gid [[thread_position_in_grid]]) {
    if (gid.x >= uint(params.outputSize.x) || gid.y >= uint(params.outputSize.y)) return;
    float globalTime = params.time * params.param0 * 1.75; float4 result = mr_layer(float2(gid), params.outputSize, globalTime, 0.0) + mr_layer(float2(gid), params.outputSize, globalTime, 0.05);
    float noiseL = smoothstep(0.3, 0.7, mr_rand(floor((float2(gid) / params.outputSize) * 64.0) / 64.0 + float2(floor(globalTime * 0.5) * 0.017, 0.0)));
    result = result * noiseL + 0.22 * float4(0.0, noiseL * 0.4, 0.0, 1.0); if (result.b < 0.5) result.b = result.g * 0.5;
    output.write(float4(clamp(result.rgb, 0.0, 1.0), 1.0), gid);
}

// MARK: - Hexagon Raymarch
static float3 hr_hue(float3 color, float shift) {
    float3 yIQ = float3(dot(color, float3(0.299, 0.587, 0.114)), dot(color, float3(0.596, -0.275, -0.321)), dot(color, float3(0.212, -0.523, 0.311)));
    float hue = atan2(yIQ.z, yIQ.y) + shift; float chroma = sqrt(yIQ.y * yIQ.y + yIQ.z * yIQ.z); yIQ.z = chroma * sin(hue); yIQ.y = chroma * cos(hue);
    color.r = dot(yIQ, float3(1.0, 0.956, 0.621)); color.g = dot(yIQ, float3(1.0, -0.272, -0.647)); color.b = dot(yIQ, float3(1.0, -1.107, 1.704)); return color;
}
static float hr_sdHexPrism(float3 p, float2 h) { float3 q = abs(p); return max(q.z - h.y, max((q.x * 0.866025 + q.y * 0.5), q.y) - h.x); }
static float2 hr_opU(float2 d1, float2 d2) { return (d1.x < d2.x) ? d1 : d2; }
static float2 hr_map(float3 pos, float iTime) {
    float height = 0.42; float depth = 0.75; float t = 0.02 + sin(iTime) * 0.01; pos.z = glslmod(pos.z, depth * 2.0) - 0.5 * depth * 2.0;
    float2 final = float2(max(-hr_sdHexPrism(pos, float2(height - t * 2.0, depth + t + 0.001)), hr_sdHexPrism(pos, float2(height - t, depth + t))), 1.5);
    for (int i = 1; i < 3; i++) { height -= 0.1; depth -= 0.19; final = hr_opU(final, float2(max(-hr_sdHexPrism(pos, float2(height - t * 2.0, depth + t + 0.001)), hr_sdHexPrism(pos, float2(height - t, depth + t))), 2.5)); } return final;
}
kernel void hexagonRaymarchKernel(texture2d<float, access::write> output [[texture(0)]], constant UniversalVisualizerParams& params [[buffer(0)]], uint2 gid [[thread_position_in_grid]]) {
    if (gid.x >= uint(params.outputSize.x) || gid.y >= uint(params.outputSize.y)) return;
    float iTime = params.time * params.param0; float2 p = -1.0 + 2.0 * (float2(gid) / params.outputSize); p.x *= params.outputSize.x / params.outputSize.y;
    float3 ro = float3(0.0, 0.0, iTime); float3 cw = normalize(float3(0.0, 0.0, 1.0)); float3 cp = float3(sin(1.570795), cos(1.570795), 0.0);
    float3 cu = normalize(cross(cw, cp)); float3 cv = normalize(cross(cu, cw)); float3 rd = normalize(p.x * cu + p.y * cv + 4.5 * cw);
    float t = 0.0, m = -1.0; float3 col = float3(1.0);
    for (int i = 0; i < 100; i++) { float2 res = hr_map(ro + rd * t, iTime); if (t > 100.0) break; t += res.x; m = res.y; }
    if ((t > 100.0 ? -1.0 : m) > -0.5) {
        float3 pos = ro + t * rd; float2 eps = float2(0.01, 0.0);
        float3 nor = normalize(float3(hr_map(pos + eps.xyy, iTime).x - hr_map(pos - eps.xyy, iTime).x, hr_map(pos + eps.yxy, iTime).x - hr_map(pos - eps.yxy, iTime).x, hr_map(pos + eps.yyx, iTime).x - hr_map(pos - eps.yyx, iTime).x));
        float occ = 0.0, sca = 1.0; for (int i = 0; i < 5; i++) { float hr = 0.01 + 0.12 * float(i) / 4.0; occ += -(hr_map(nor * hr + pos, iTime).x - hr) * sca; sca *= 0.95; }
        col = 1.0 - hr_hue(float3(0.0, 1.0, 1.0), iTime * 0.02 + pos.z) * clamp(1.0 - 3.0 * occ, 0.0, 1.0);
    }
    output.write(float4(clamp(col, 0.0, 1.0), 1.0), gid);
}

// MARK: - Octagrams
static float2x2 og_rot(float a) { float c = cos(a), s = sin(a); return float2x2(float2(c, s), float2(-s, c)); }
static float og_box(float3 pos, float scale) { pos *= scale; float3 q = abs(pos) - float3(0.4, 0.4, 0.1); float base = (length(max(q, 0.0)) + min(max(q.x, max(q.y, q.z)), 0.0)) / 1.5; pos.xy *= 5.0; pos.y -= 3.5; pos.xy = og_rot(0.75) * pos.xy; return -base; }
static float og_box_set(float3 pos, float gTime) {
    float sinG = sin(gTime * 0.4); float scale = 2.0 - abs(sinG) * 1.5; float3 po = pos;
    pos = po; pos.y += sinG * 2.5; pos.xy = og_rot(0.8) * pos.xy; float box1 = og_box(pos, scale);
    pos = po; pos.y -= sinG * 2.5; pos.xy = og_rot(0.8) * pos.xy; float box2 = og_box(pos, scale);
    pos = po; pos.x += sinG * 2.5; pos.xy = og_rot(0.8) * pos.xy; float box3 = og_box(pos, scale);
    pos = po; pos.x -= sinG * 2.5; pos.xy = og_rot(0.8) * pos.xy; float box4 = og_box(pos, scale);
    pos = po; pos.xy = og_rot(0.8) * pos.xy; float box5 = og_box(pos, 0.5) * 6.0;
    return max(max(max(max(max(box1, box2), box3), box4), box5), og_box(po, 0.5) * 6.0);
}
kernel void octagramsKernel(texture2d<float, access::write> output [[texture(0)]], constant UniversalVisualizerParams& params [[buffer(0)]], uint2 gid [[thread_position_in_grid]]) {
    if (gid.x >= uint(params.outputSize.x) || gid.y >= uint(params.outputSize.y)) return;
    float iTime = params.time * params.param0; float3 ray = normalize(float3((float2(gid) * 2.0 - params.outputSize) / min(params.outputSize.x, params.outputSize.y), 1.5));
    ray.xy = og_rot(sin(iTime * 0.03) * 5.0) * ray.xy; ray.yz = og_rot(sin(iTime * 0.05) * 0.2) * ray.yz;
    float t = 0.1, ac = 0.0;
    for (int i = 0; i < 99; i++) { float d = max(abs(og_box_set(glslmod3f(float3(0.0, -0.2, iTime * 4.0) + ray * t - 2.0, 4.0) - 2.0, iTime - float(i) * 0.01)), 0.01); ac += exp(-d * 23.0); t += d * 0.55; }
    output.write(float4(float3(ac * 0.02) + float3(0.0, 0.2 * abs(sin(iTime)), 0.5 + sin(iTime) * 0.2), 1.0), gid);
}

// MARK: - Pink Warp FBM
static float pw_rand(float2 n) { return fract(sin(dot(n, float2(12.9898, 4.1414))) * 43758.5453); }
static float pw_fbm(float2 p, float iTime) {
    float f = 0.0; float2 ip, u; float2x2 mtx = float2x2(float2(0.80, 0.60), float2(-0.60, 0.80));
    for (int i = 0; i < 6; i++) {
        ip = floor(p + (i == 5 ? sin(iTime) : (i == 0 ? iTime : 0.0))); u = fract(p + (i == 5 ? sin(iTime) : (i == 0 ? iTime : 0.0))); u = u * u * (3.0 - 2.0 * u);
        f += (i == 0 ? 0.5 : (i == 1 ? 0.03125 : (i == 2 ? 0.25 : (i == 3 ? 0.125 : (i == 4 ? 0.0625 : 0.015625))))) * mix(mix(pw_rand(ip), pw_rand(ip + float2(1.0, 0.0)), u.x), mix(pw_rand(ip + float2(0.0, 1.0)), pw_rand(ip + float2(1.0, 1.0)), u.x), u.y);
        p = mtx * p * (i == 4 ? 2.04 : (i == 3 || i == 1 ? 2.01 : (i == 2 ? 2.03 : 2.02)));
    } return f / 0.96875;
}
kernel void pinkWarpFBMKernel(texture2d<float, access::write> output [[texture(0)]], constant UniversalVisualizerParams& params [[buffer(0)]], uint2 gid [[thread_position_in_grid]]) {
    if (gid.x >= uint(params.outputSize.x) || gid.y >= uint(params.outputSize.y)) return;
    float x = pw_fbm(float2(gid) / params.outputSize.x + pw_fbm(float2(gid) / params.outputSize.x + pw_fbm(float2(gid) / params.outputSize.x, params.time * params.param0), params.time * params.param0), params.time * params.param0);
    output.write(float4(x < 0.0 ? 54.0 / 255.0 : (x < 20049.0 / 82979.0 ? (829.79 * x + 54.51) / 255.0 : 1.0),
                        x < 20049.0 / 82979.0 ? 0.0 : (x < 327013.0 / 810990.0 ? (8546482679670.0 / 10875673217.0 * x - 2064961390770.0 / 10875673217.0) / 255.0 : (x <= 1.0 ? (103806720.0 / 483977.0 * x + 19607415.0 / 483977.0) / 255.0 : 1.0)),
                        x < 0.0 ? 54.0 / 255.0 : (x < 7249.0 / 82979.0 ? (829.79 * x + 54.51) / 255.0 : (x < 20049.0 / 82979.0 ? 127.0 / 255.0 : (x < 327013.0 / 810990.0 ? (792.02249341361 * x - 64.36479073560) / 255.0 : 1.0))), 1.0), gid);
}

// MARK: - Blue Crumpled Wave
kernel void blueCrumpledWaveKernel(texture2d<float, access::write> output [[texture(0)]], constant UniversalVisualizerParams& params [[buffer(0)]], uint2 gid [[thread_position_in_grid]]) {
    if (gid.x >= uint(params.outputSize.x) || gid.y >= uint(params.outputSize.y)) return;
    float2 uv = (2.0 * float2(gid) - params.outputSize) / min(params.outputSize.x, params.outputSize.y);
    for (float i = 1.0; i < 8.0; i++) uv.y += i * 0.1 / i * sin(uv.x * i * i + params.time * params.param0 * 0.5) * sin(uv.y * i * i + params.time * params.param0 * 0.5);
    output.write(float4(uv.y - 0.1, uv.y + 0.3, uv.y + 0.95, 1.0), gid);
}

// MARK: - Golden Wave Vortex
static float2x2 gv_r(float a) { float c = cos(a), s = sin(a); return float2x2(float2(c, s), float2(-s, c)); }
static float4 gv_tonemap(float4 o) { float l = dot(o.rgb, float3(0.2126, 0.7152, 0.0722)); return l < 1e-6 ? o : float4(o.rgb * (l * (1.0 + l / 4.0) / (1.0 + l)) / l, o.a); }
static float gv_n(float3 p) {
    const float3x3 G = float3x3(float3(-0.571464913, +0.814921382, +0.096597072), float3(-0.278044873, -0.303026659, +0.911518454), float3(+0.772087367, +0.494042493, +0.399753815));
    return dot(cos(G * p), sin(1.618033988 * p * G));
}
kernel void goldenWaveVortexKernel(texture2d<float, access::write> output [[texture(0)]], constant UniversalVisualizerParams& params [[buffer(0)]], uint2 gid [[thread_position_in_grid]]) {
    if (gid.x >= uint(params.outputSize.x) || gid.y >= uint(params.outputSize.y)) return;
    float t = params.time * params.param0; float3 p = float3(0.0, 0.0, -t * 0.9); float3 d = normalize(float3(2.0 * float2(gid) - params.outputSize, params.outputSize.y)); float4 o = float4(0.0);
    for (int i = 0; i < 10; i++) { float3 b = p; b.xy = gv_r(t * 0.1 + b.z * 0.2) * b.xy; float sv = max(abs(gv_n(p * 0.1) / 5.0 - gv_n(b * 0.9)), (10.0 - length(b.xy) * 1.1)) + abs(sin(p.z + t)) * 0.5; p += d * sv; o += (1.0 + sin(float(i) * 0.1 + length(b.xy * 0.05 + 9.0) + float4(3.0, 2.5, 1.0, 1.0) + (float(i) * 100.0))) / sv; }
    output.write(float4(clamp(gv_tonemap(o / 30.0).rgb, 0.0, 1.0), 1.0), gid);
}

// MARK: - Neon Rectangles
kernel void neonRectanglesKernel(texture2d<float, access::write> output [[texture(0)]], constant UniversalVisualizerParams& params [[buffer(0)]], uint2 gid [[thread_position_in_grid]]) {
    if (gid.x >= uint(params.outputSize.x) || gid.y >= uint(params.outputSize.y)) return;
    float2 p = (float2(gid) - params.outputSize * 0.5) / (params.outputSize.y * 0.5); float3 col = float3(0.0);
    for (float i = 0.0; i < max(params.param1 > 0.0 ? params.param1 : 8.0, 1.0); i++) {
        float phase = fract(i / max(params.param1 > 0.0 ? params.param1 : 8.0, 1.0) + fract(params.time * params.param0 * 0.18)); float2 d = abs(p) - float2(mix(0.05, 1.05, phase) * params.outputSize.x / params.outputSize.y, mix(0.05, 1.05, phase));
        float dist = abs(length(max(d, 0.0)) + min(max(d.x, d.y), 0.0)); float baseT = (params.param2 > 0.0 ? params.param2 : 1.0) * 1.4;
        col += (exp(-dist * 220.0 / baseT) * 1.3 + exp(-dist * 22.0 / baseT) * 0.8 + exp(-dist * 6.0 / baseT) * 0.3) * smoothstep(0.0, 0.15, phase) * smoothstep(1.0, 0.7, phase) * max(mix(float3(max(pow(max(0.0, cos(6.28318 * (fract(i / max(params.param1 > 0.0 ? params.param1 : 8.0, 1.0) + params.time * params.param0 * 0.08) + 0.00)) * 0.5 + 0.5), 0.5), max(pow(max(0.0, cos(6.28318 * (fract(i / max(params.param1 > 0.0 ? params.param1 : 8.0, 1.0) + params.time * params.param0 * 0.08) + 0.33)) * 0.5 + 0.5), 0.5), pow(max(0.0, cos(6.28318 * (fract(i / max(params.param1 > 0.0 ? params.param1 : 8.0, 1.0) + params.time * params.param0 * 0.08) + 0.67)) * 0.5 + 0.5), 0.5)))), float3(pow(max(0.0, cos(6.28318 * (fract(i / max(params.param1 > 0.0 ? params.param1 : 8.0, 1.0) + params.time * params.param0 * 0.08) + 0.00)) * 0.5 + 0.5), 0.5), pow(max(0.0, cos(6.28318 * (fract(i / max(params.param1 > 0.0 ? params.param1 : 8.0, 1.0) + params.time * params.param0 * 0.08) + 0.33)) * 0.5 + 0.5), 0.5), pow(max(0.0, cos(6.28318 * (fract(i / max(params.param1 > 0.0 ? params.param1 : 8.0, 1.0) + params.time * params.param0 * 0.08) + 0.67)) * 0.5 + 0.5), 0.5)), 2.2), float3(0.0));
    }
    float3 q = fract(float3(p * 97.3, params.time * params.param0 * 3.7)); q += dot(q, q.yzx + 33.33);
    output.write(float4(pow(1.0 - exp(-col * 1.3), float3(0.85, 0.9, 1.0)) * (fract((q.x + q.y) * q.z) * 0.1 + 0.9), 1.0), gid);
}

