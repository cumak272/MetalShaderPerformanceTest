//
//  MetalDeviceProvider.swift
//  MetalCanvasTest
//
//  Created by Cuma Ali Kesici on 16.02.26.
//

import MetalKit

enum MetalDeviceProvider {
    static let device: MTLDevice = {
        guard let device = MTLCreateSystemDefaultDevice() else {
            fatalError("Metal is not supported on this device")
        }
        return device
    }()

    static let commandQueue: MTLCommandQueue = {
        guard let queue = device.makeCommandQueue() else {
            fatalError("Could not create Metal command queue")
        }
        return queue
    }()
}
