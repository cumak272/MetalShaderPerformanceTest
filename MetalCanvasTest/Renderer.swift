//
//  Renderer.swift
//  MetalCanvasTest
//
//  Created by Cuma Ali Kesici on 16.02.26.
//

import MetalKit
import QuartzCore

class Renderer: NSObject, MTKViewDelegate {

    let device: MTLDevice = MetalDeviceProvider.device
    let commandQueue: MTLCommandQueue = MetalDeviceProvider.commandQueue
    var computePipelineState: MTLComputePipelineState!
    var visualizer: any Visualizer {
        didSet {
            setupPipeline()
        }
    }
    let startTime: CFTimeInterval = CACurrentMediaTime()

    private let frameSemaphore = DispatchSemaphore(value: 3)

    init(visualizer: any Visualizer) {
        self.visualizer = visualizer
        super.init()
        setupPipeline()
    }

    func setupPipeline() {
        guard let library = device.makeDefaultLibrary() else {
            print("Could not find default library")
            return
        }

        guard let kernelFunction = library.makeFunction(name: visualizer.kernelName) else {
            print("Could not find kernel function: \(visualizer.kernelName)")
            return
        }

        do {
            computePipelineState = try device.makeComputePipelineState(function: kernelFunction)
        } catch {
            print("Error creating compute pipeline state: \(error)")
        }
    }

    func mtkView(_ view: MTKView, drawableSizeWillChange size: CGSize) {

    }

    func draw(in view: MTKView) {

        frameSemaphore.wait()

        guard let drawable = view.currentDrawable,
              let computePipelineState = computePipelineState else {
            frameSemaphore.signal()
            return
        }

        let commandBuffer = commandQueue.makeCommandBuffer()!

        commandBuffer.addCompletedHandler { [weak self] _ in
            self?.frameSemaphore.signal()
        }

        let commandEncoder = commandBuffer.makeComputeCommandEncoder()!
        commandEncoder.setComputePipelineState(computePipelineState)
        commandEncoder.setTexture(drawable.texture, index: 0)

        let time = Float(CACurrentMediaTime() - startTime)
        var params = visualizer.params(time: time, size: view.drawableSize)

        commandEncoder.setBytes(&params, length: MemoryLayout<UniversalVisualizerParams>.stride, index: 0)

        let w = computePipelineState.threadExecutionWidth
        let h = computePipelineState.maxTotalThreadsPerThreadgroup / w
        let threadsPerThreadgroup = MTLSizeMake(w, h, 1)

        let threadsPerGrid = MTLSizeMake(Int(view.drawableSize.width), Int(view.drawableSize.height), 1)

        commandEncoder.dispatchThreads(threadsPerGrid, threadsPerThreadgroup: threadsPerThreadgroup)
        commandEncoder.endEncoding()

        commandBuffer.present(drawable)
        commandBuffer.commit()
    }
}
