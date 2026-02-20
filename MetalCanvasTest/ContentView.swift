//
//  ContentView.swift
//  MetalCanvasTest
//
//  Created by Cuma Ali Kesici on 16.02.26.
//

import SwiftUI
import MetalKit

struct ContentView: UIViewRepresentable {
    @ObservedObject var visualizerManager = VisualizerManager.shared

    func makeCoordinator() -> Renderer {
        Renderer(visualizer: visualizerManager.selectedVisualizer ?? visualizerManager.visualizers.first!)
    }

    func makeUIView(context: UIViewRepresentableContext<ContentView>) -> MTKView {

        let mtkView = MTKView()
        mtkView.device = MetalDeviceProvider.device
        mtkView.delegate = context.coordinator
        mtkView.preferredFramesPerSecond = 60
        mtkView.enableSetNeedsDisplay = false
        mtkView.isPaused = false
        mtkView.framebufferOnly = false

        return mtkView
    }

    func updateUIView(_ uiView: MTKView, context: UIViewRepresentableContext<ContentView>) {
        if let selectedVisualizer = visualizerManager.selectedVisualizer,
           context.coordinator.visualizer.name != selectedVisualizer.name {
            context.coordinator.visualizer = selectedVisualizer
        }
    }
}

struct ContentView_Previews: PreviewProvider {
    static var previews: some View {
        ContentView()
    }
}
