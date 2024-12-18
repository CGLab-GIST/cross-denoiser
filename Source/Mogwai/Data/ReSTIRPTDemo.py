from falcor import *
import os

out_dir = 'results'

def add_capture(g, pairs, start=0, end=100000, opts=None):
    loadRenderPassLibrary("CapturePass.dll")

    channels = list(pairs.keys())
    inputs = list(pairs.values())

    options = {
            'directory': out_dir,
            'channels': channels,
            'exitAtEnd': True,
            'accumulate': False,
            'writeStart': start,    # Control frame number in below
            'writeEnd': end,  # Control frame number in below
            'captureCameraMat': False,
            'includeAlpha': ["specRough", "diffuseOpacity", "specRough2", "diffuseOpacity2"],
    }
    if opts is not None:
        options.update(opts)
    CapturePass = createPass("CapturePass", options)

    capture = "CapturePass"
    g.addPass(CapturePass, capture)

    def addEdgeOutput(input, channel):
        g.addEdge(input, f"{capture}.{channel}")
        g.markOutput(f"{capture}.{channel}")

    for input, channel in zip(inputs, channels):
        addEdgeOutput(input, channel)

    return capture


def render_graph_ReSTIRPT(gvbuffer):
    g = RenderGraph("ReSTIRPTPass")
    loadRenderPassLibrary("AccumulatePass.dll")
    loadRenderPassLibrary("GBuffer.dll")
    loadRenderPassLibrary("ReSTIRPTPass.dll")
    loadRenderPassLibrary("ToneMapper.dll")
    loadRenderPassLibrary("ScreenSpaceReSTIRPass.dll")
    loadRenderPassLibrary("ErrorMeasurePass.dll")
    loadRenderPassLibrary("ImageLoader.dll")

    ReSTIRGIPlusPass = createPass("ReSTIRPTPass", {'samplesPerPixel': 1})
    g.addPass(ReSTIRGIPlusPass, "ReSTIRPTPass")
    GVBuffer = createPass(f"{gvbuffer}", {'samplePattern': SamplePattern.Center, 'sampleCount': 1, 'texLOD': TexLODMode.Mip0, 'useAlphaTest': True})
    g.addPass(GVBuffer, f"{gvbuffer}")
    AccumulatePass = createPass("AccumulatePass", {'enableAccumulation': False, 'precisionMode': AccumulatePrecision.Double})
    g.addPass(AccumulatePass, "AccumulatePass")
    ToneMapper = createPass("ToneMapper", {'autoExposure': False, 'exposureCompensation': 0.0, 'operator': ToneMapOp.Linear})
    g.addPass(ToneMapper, "ToneMapper")
    ScreenSpaceReSTIRPass = createPass("ScreenSpaceReSTIRPass")
    g.addPass(ScreenSpaceReSTIRPass, "ScreenSpaceReSTIRPass")

    g.addEdge(f"{gvbuffer}.vbuffer", "ReSTIRPTPass.vbuffer")
    g.addEdge(f"{gvbuffer}.mvec", "ReSTIRPTPass.motionVectors")

    g.addEdge(f"{gvbuffer}.vbuffer", "ScreenSpaceReSTIRPass.vbuffer")
    g.addEdge(f"{gvbuffer}.mvec", "ScreenSpaceReSTIRPass.motionVectors")
    g.addEdge("ScreenSpaceReSTIRPass.color", "ReSTIRPTPass.directLighting")

    g.addEdge("ReSTIRPTPass.color", "AccumulatePass.input")
    g.addEdge("AccumulatePass.output", "ToneMapper.src")

    g.markOutput("ToneMapper.dst")
    g.markOutput("AccumulatePass.output")

    return g

def add_SVGF(g, gvbuffer):
    loadRenderPassLibrary("SVGFPass.dll")
    SVGFPass = createPass("SVGFPass", {'Enabled': True, 'Iterations': 4, 'FeedbackTap': 1, 'VarianceEpsilon': 9.999999747378752e-05, 'PhiColor': 10.0, 'PhiNormal': 128.0, 'Alpha': 0.05000000074505806, 'MomentsAlpha': 0.20000000298023224})
    g.addPass(SVGFPass, "SVGFPass")
    g.addEdge("ReSTIRPTPass.color", "SVGFPass.Color")
    g.addEdge("ReSTIRPTPass.albedo", "SVGFPass.Albedo")
    g.addEdge(f"{gvbuffer}.emissive", "SVGFPass.Emission")
    g.addEdge(f"{gvbuffer}.posW", "SVGFPass.WorldPosition")
    g.addEdge(f"{gvbuffer}.normW", "SVGFPass.WorldNormal")
    g.addEdge(f"{gvbuffer}.pnFwidth", "SVGFPass.PositionNormalFwidth")
    g.addEdge(f"{gvbuffer}.linearZ", "SVGFPass.LinearZ")
    g.addEdge(f"{gvbuffer}.mvec", "SVGFPass.MotionVec")
    g.markOutput("SVGFPass.Filtered image")

    return "SVGFPass"

gvbuffer = "GBufferRaster" # Enable SVGF
# gvbuffer = "VBufferRT"
graph_ReSTIRPT = render_graph_ReSTIRPT(gvbuffer)
if gvbuffer == "GBufferRaster":
    add_SVGF(graph_ReSTIRPT, gvbuffer)
capture_pairs = {
    'color': 'ReSTIRPTPass.color',
    'albedo': 'ReSTIRPTPass.albedo',
    'envLight': 'ReSTIRPTPass.envLight',
    'emissive': f'{gvbuffer}.emissive',
    'denoised': 'SVGFPass.Filtered image',
}
add_capture(graph_ReSTIRPT, capture_pairs)

m.addGraph(graph_ReSTIRPT)
m.loadScene('C:/Users/hchoi/repositories//ORCA/Bistro/BistroExterior2.pyscene')
# Call this after scene loading to change the default camera
m.scene.camera.nearPlane = 0.15 # Increase near plane to prevent Z-fighting in the rasterizer

################################################################################
# Rendering
m.clock.framerate = 30
m.clock.time = 0
m.clock.pause()

for frame in range(0, 10):
    m.clock.frame = frame
    m.renderFrame()

exit()



