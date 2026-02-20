//
//  ExternalDisplayManager.swift
//  MetalCanvasTest
//
//  Created by Cuma Ali Kesici on 16.02.26.
//

import UIKit
import Combine

class ExternalDisplayManager: ObservableObject {
    static let shared = ExternalDisplayManager()
    
    @Published var isExternalDisplayConnected: Bool = false
    @Published var additionalWindows: [UIWindow] = []
    
    private init() {}
    
    func externalDisplayDidConnect(screen: UIScreen, window: UIWindow) {
        self.additionalWindows.append(window)
        self.isExternalDisplayConnected = true
        print("External display connected: \(screen)")
    }
    
    func externalDisplayDidDisconnect() {
        self.additionalWindows.removeAll()
        self.isExternalDisplayConnected = false
        print("External display disconnected")
    }
}
