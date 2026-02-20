//
//  Visualizer.swift
//  MetalCanvasTest
//
//  Created by Cuma Ali Kesici on 16.02.26.
//

import Foundation
import Combine
import SwiftUI

struct UniversalVisualizerParams {
    var time: Float
    var outputSize: SIMD2<Float>
    var param0: Float
    var param1: Float
    var param2: Float
    var param3: Float
    var param4: Float
    var param5: Float
    
    init(time: Float, outputSize: SIMD2<Float>, param0: Float = 0, param1: Float = 0, param2: Float = 0, param3: Float = 0, param4: Float = 0, param5: Float = 0) {
        self.time = time
        self.outputSize = outputSize
        self.param0 = param0
        self.param1 = param1
        self.param2 = param2
        self.param3 = param3
        self.param4 = param4
        self.param5 = param5
    }
}

protocol Visualizer {
    var name: String { get }
    var kernelName: String { get }
    func params(time: Float, size: CGSize) -> UniversalVisualizerParams
}

struct DefaultVisualizer: Visualizer {
    var name: String
    var kernelName: String
    var param0: Float = 0
    var param1: Float = 0
    var param2: Float = 0
    var param3: Float = 0
    var param4: Float = 0
    var param5: Float = 0
    
    func params(time: Float, size: CGSize) -> UniversalVisualizerParams {
        return UniversalVisualizerParams(time: time, outputSize: SIMD2<Float>(Float(size.width), Float(size.height)), param0: param0, param1: param1, param2: param2, param3: param3, param4: param4, param5: param5)
    }
}

class VisualizerManager: ObservableObject {
    static let shared = VisualizerManager()
    
    @Published var visualizers: [any Visualizer] = [
        DefaultVisualizer(name: "Blue Fragments", kernelName: "blueFragmentsKernel", param0: 1.0),
        DefaultVisualizer(name: "Solid Color", kernelName: "solidColorGeneratorKernel", param0: 1.0, param1: 0.0, param2: 0.0, param3: 1.0),
        DefaultVisualizer(name: "Plasma", kernelName: "plasmaGeneratorKernel", param0: 1.0),
        DefaultVisualizer(name: "Color Wheel", kernelName: "colorWheelGeneratorKernel", param0: 1.0),
        DefaultVisualizer(name: "Psychedelics", kernelName: "psychedelicsGeneratorKernel", param0: 1.0),
        DefaultVisualizer(name: "Lightspeed", kernelName: "lightspeedGeneratorKernel", param0: 150.0, param1: 80.0, param2: 45.0),
        DefaultVisualizer(name: "Fractal Pattern", kernelName: "fractalPatternGeneratorKernel", param0: 10.0, param1: 3.0, param2: 2.0),
        DefaultVisualizer(name: "Mandelbrot", kernelName: "mandelbrotGeneratorKernel", param0: 1.0),
        DefaultVisualizer(name: "Apollonian", kernelName: "apollonianGeneratorKernel", param0: 1.0, param1: 1.5, param2: 0.33),
        DefaultVisualizer(name: "Rick and Morty Portal", kernelName: "rickAndMortyPortalGeneratorKernel", param0: 1.0),
        DefaultVisualizer(name: "Neon Rings", kernelName: "neonRingsGeneratorKernel", param0: 1.0, param1: 5.0, param2: 0.05),
        DefaultVisualizer(name: "Plasma Storm", kernelName: "plasmaStormGeneratorKernel", param0: 1.0, param1: 4.0, param2: 0.5, param3: 1.5, param4: 1.2),
        DefaultVisualizer(name: "Cosmic Smoke", kernelName: "cosmicSmokeKernel", param0: 1.0),
        DefaultVisualizer(name: "Menger Tunnel", kernelName: "mengerTunnelKernel", param0: 1.0),
        DefaultVisualizer(name: "Sponge Tunnel", kernelName: "spongeTunnelKernel", param0: 1.0),
        DefaultVisualizer(name: "Kaleidoscope Tunnel", kernelName: "kaleidoscopeTunnelKernel", param0: 1.0),
        DefaultVisualizer(name: "Warp Bump", kernelName: "warpBumpKernel", param0: 1.0, param1: 1.5),
        DefaultVisualizer(name: "Cave Tunnel", kernelName: "caveTunnelKernel", param0: 1.0),
        DefaultVisualizer(name: "Monster", kernelName: "monsterKernel", param0: 1.0),
        DefaultVisualizer(name: "Phantom Star", kernelName: "phantomStarKernel", param0: 1.0),
        DefaultVisualizer(name: "Glowing Marble", kernelName: "glowingMarbleKernel", param0: 1.0),
        DefaultVisualizer(name: "Hex Truchet", kernelName: "hexTruchetKernel", param0: 1.0),
        DefaultVisualizer(name: "Matrix Rain", kernelName: "matrixRainKernel", param0: 1.0),
        DefaultVisualizer(name: "Hexagon Raymarch", kernelName: "hexagonRaymarchKernel", param0: 1.0),
        DefaultVisualizer(name: "Octagrams", kernelName: "octagramsKernel", param0: 1.0),
        DefaultVisualizer(name: "Pink Warp FBM", kernelName: "pinkWarpFBMKernel", param0: 1.0),
        DefaultVisualizer(name: "Blue Crumpled Wave", kernelName: "blueCrumpledWaveKernel", param0: 1.0),
        DefaultVisualizer(name: "Golden Wave Vortex", kernelName: "goldenWaveVortexKernel", param0: 1.0),
        DefaultVisualizer(name: "Neon Rectangles", kernelName: "neonRectanglesKernel", param0: 1.0, param1: 8.0, param2: 1.0)
    ]
    
    @Published var selectedVisualizer: (any Visualizer)? = nil
    
    init() {
        selectedVisualizer = visualizers.first
    }
}
