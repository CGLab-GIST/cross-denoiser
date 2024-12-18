# type: ignore
import os

FILE = "Bistro/BistroExterior.pyscene"
INTERACTIVE = True
OUT_DIR = "output"
NAME = "Bistro"
ANIM = [1400, 1410]

def add_path(g, gbuf, enable_restir=True, samples_per_pixel=1):
    loadRenderPassLibrary("ReSTIRPTPass.dll")
    loadRenderPassLibrary("ScreenSpaceReSTIRPass.dll")

    if enable_restir:
        PathTracer = createPass("ReSTIRPTPass", {
            'samplesPerPixel': samples_per_pixel,
            'temporalSeedOffset': 0,
        })
        path = "ReSTIRPT"
        ScreenSpaceReSTIRPass = createPass("ScreenSpaceReSTIRPass", {
            'NumReSTIRInstances': samples_per_pixel,
        })
        screenReSTIR = "ScreenSpaceReSTIR"
    else:
        PathTracer = createPass("ReSTIRPTPass", {
            'samplesPerPixel': samples_per_pixel,
            'pathSamplingMode': PathSamplingMode.PathTracing,
        })
        path = "ReSTIRPT"
        ScreenSpaceReSTIRPass = createPass("ScreenSpaceReSTIRPass", {
            'NumReSTIRInstances': samples_per_pixel,
            'options':ScreenSpaceReSTIROptions(
                useTemporalResampling=False, useSpatialResampling=False,
            )
        })
        screenReSTIR = "ScreenSpaceReSTIR"

    g.addPass(PathTracer, path)
    g.addPass(ScreenSpaceReSTIRPass, screenReSTIR)

    g.addEdge(f"{gbuf}.vbuffer", f"{path}.vbuffer")
    g.addEdge(f"{gbuf}.mvec", f"{path}.motionVectors")
    g.addEdge(f"{gbuf}.vbuffer", f"{screenReSTIR}.vbuffer")
    g.addEdge(f"{gbuf}.mvec", f"{screenReSTIR}.motionVectors")
    g.addEdge(f"{screenReSTIR}.color", f"{path}.directLighting")
    g.addEdge(f"{screenReSTIR}.color2", f"{path}.directLighting2")

    return path, screenReSTIR

def add_gbuffer(g):
    loadRenderPassLibrary("GBuffer.dll")

    dicts = {
        'samplePattern': SamplePattern.Center,
        'useAlphaTest': True,
    }

    GBuffer = createPass("GBufferRaster", dicts)
    gbuf = "GBufferRaster"

    g.addPass(GBuffer, gbuf)
    return gbuf

def add_capture(g, pairs, start, end, opts=None):
    loadRenderPassLibrary("CapturePass.dll")

    channels = list(pairs.keys())
    inputs = list(pairs.values())

    options = {
        'directory': OUT_DIR,
        'channels': channels,
        'accumulate': False,
        'captureCameraMat': False,
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

def add_torch(g):
    loadRenderPassLibrary("TorchPass.dll")

    TorchPass = createPass("TorchPass", {})
    torch = "TorchPass"

    g.addPass(TorchPass, torch)
    return torch

def render_torch(start, end):
    g = RenderGraph("MutlipleGraph")

    ## GBufferRaster
    gbuf = add_gbuffer(g)

    ## PathTracer
    path, ss_restir = add_path(g, gbuf, enable_restir=True, samples_per_pixel=2)

    ## TorchPass
    torch = add_torch(g)

    g.addEdge(f"{path}.color", f"{torch}.color")
    g.addEdge(f"{path}.color2", f"{torch}.color2")
    g.addEdge(f"{gbuf}.texC", f"{torch}.albedo")
    g.addEdge(f"{gbuf}.normW", f"{torch}.normal")
    g.addEdge(f"{gbuf}.mvec", f"{torch}.mvec")
    g.addEdge(f"{gbuf}.pnFwidth", f"{torch}.pnFwidth")
    g.addEdge(f"{gbuf}.linearZ", f"{torch}.linearZ")
    g.addEdge(f"{gbuf}.diffuseOpacity", f"{torch}.diffuseOpacity")

    # ToneMapper for visualization
    loadRenderPassLibrary("ToneMapper.dll")
    if NAME == "Bistro":
        exp = 1.5
    elif NAME == "EmeraldSquare":
        exp = 1.0
    else:
        exp = 0.0
    ToneMapper = createPass("ToneMapper", {'autoExposure': False, 'exposureValue': exp})
    g.addPass(ToneMapper, "ToneMapper")
    g.addEdge(f"{torch}.out", "ToneMapper.src")
    g.markOutput("ToneMapper.dst")

    g.markOutput(f"{torch}.out")
    g.markOutput(f"{torch}.crossA")
    g.markOutput(f"{torch}.crossB")
    g.markOutput(f"{path}.color")
    g.markOutput(f"{path}.color2")

    if not INTERACTIVE:
        pairs = {
            'crossA': f"{torch}.crossA",
            'crossB': f"{torch}.crossB",
            'out': f"{torch}.out",
        }
        add_capture(g, pairs, start, end, {'captureCameraMat': False})

    return g

print("ANIM = ", ANIM)
graph = render_torch(*ANIM)

m.addGraph(graph)
m.loadScene(FILE)
# Call this after scene loading
m.scene.camera.nearPlane = 0.15 # Increase near plane to prevent Z-fighting

m.clock.framerate = 60
m.clock.time = 0
m.clock.frame = ANIM[0]

if not INTERACTIVE:
    num_frames = ANIM[1] - ANIM[0] + 1

    # Start frame
    for frame in range(num_frames):
        m.clock.frame = ANIM[0] + frame
        print('Rendering frame:', m.clock.frame)
        m.renderFrame()

    exit(0)
