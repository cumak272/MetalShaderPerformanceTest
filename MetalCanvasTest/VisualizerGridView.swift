//
//  VisualizerGridView.swift
//  MetalCanvasTest
//
//  Created by Cuma Ali Kesici on 16.02.26.
//

import SwiftUI
import MetalKit

struct VisualizerGridView: View {
    @StateObject var visualizerManager = VisualizerManager.shared
    
    let columns = [
        GridItem(.adaptive(minimum: 150))
    ]
    
    var body: some View {
        ScrollView {
            LazyVGrid(columns: columns, spacing: 20) {
                ForEach(visualizerManager.visualizers, id: \.name) { visualizer in
                    VStack {
                        VisualizerPreview(visualizer: visualizer)
                            .aspectRatio(1, contentMode: .fit)
                            .cornerRadius(10)
                            .overlay(
                                RoundedRectangle(cornerRadius: 10)
                                    .stroke(
                                        visualizerManager.selectedVisualizer?.name == visualizer.name ? Color.blue : Color.white,
                                        lineWidth: visualizerManager.selectedVisualizer?.name == visualizer.name ? 4 : 2
                                    )
                            )
                            .onTapGesture {
                                visualizerManager.selectedVisualizer = visualizer
                            }
                        
                        MarqueeText(text: visualizer.name, font: .headline)
                            .foregroundColor(.white)
                    }
                }
            }
            .padding()
        }
        .background(Color.black)
    }
}

struct VisualizerPreview: UIViewRepresentable {
    let visualizer: any Visualizer
    
    func makeCoordinator() -> Renderer {
        Renderer(visualizer: visualizer)
    }

    func makeUIView(context: Context) -> MTKView {
        let mtkView = MTKView()
        mtkView.device = MetalDeviceProvider.device
        mtkView.delegate = context.coordinator
        mtkView.preferredFramesPerSecond = 15
        mtkView.enableSetNeedsDisplay = false
        mtkView.isPaused = false
        mtkView.framebufferOnly = false

        return mtkView
    }
    
    func updateUIView(_ uiView: MTKView, context: Context) {
        
    }
}
//
//  MarqueeText.swift
//  MetalCanvasTest
//

import SwiftUI

struct MarqueeText: View {
    var text: String
    var font: Font
    
    @State private var offset: CGFloat = 0
    @State private var storedTextWidth: CGFloat = 0
    
    var body: some View {
        GeometryReader { geo in
            let isTooLong = storedTextWidth > geo.size.width
            
            Group {
                if isTooLong {
                    HStack(spacing: 30) {
                        Text(text)
                            .font(font)
                            .lineLimit(1)
                            .fixedSize(horizontal: true, vertical: false)
                        
                        Text(text)
                            .font(font)
                            .lineLimit(1)
                            .fixedSize(horizontal: true, vertical: false)
                    }
                    .offset(x: offset)
                    .onAppear {
                        let distance = storedTextWidth + 30
                        withAnimation(.linear(duration: Double(distance) / 30.0).delay(1.0).repeatForever(autoreverses: false)) {
                            offset = -distance
                        }
                    }
                } else {
                    Text(text)
                        .font(font)
                        .lineLimit(1)
                        .frame(width: geo.size.width, alignment: .center)
                }
            }
            .background(
                Text(text)
                    .font(font)
                    .lineLimit(1)
                    .fixedSize(horizontal: true, vertical: false)
                    .opacity(0)
                    .background(GeometryReader { textGeo in
                        Color.clear.onAppear {
                            self.storedTextWidth = textGeo.size.width
                        }
                    })
            )
        }
        .clipped()
        .frame(height: 24)
    }
}
