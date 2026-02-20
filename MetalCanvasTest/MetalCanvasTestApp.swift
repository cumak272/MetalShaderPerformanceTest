//
//  MetalCanvasTestApp.swift
//  MetalCanvasTest
//
//  Created by Cuma Ali Kesici on 16.02.26.
//

import SwiftUI

@main
struct MetalCanvasTestApp: App {
    @UIApplicationDelegateAdaptor(AppDelegate.self) var appDelegate
    
    var body: some Scene {
        WindowGroup {
            VisualizerGridView()
        }
    }
}
