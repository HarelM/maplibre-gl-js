void main() {
    // Native depth is written automatically by the GPU via gl_FragCoord.z.
    // Write it to the R32F color attachment for CPU readback in depthAtPoint().
    fragColor = vec4(gl_FragCoord.z, 0.0, 0.0, 0.0);
}
