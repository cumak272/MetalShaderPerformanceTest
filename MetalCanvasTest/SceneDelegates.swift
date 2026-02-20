//
//  SceneDelegates.swift
//  MetalCanvasTest
//
//  Created by Cuma Ali Kesici on 16.02.26.
//

import UIKit
import SwiftUI

class ExternalDisplaySceneDelegate: UIResponder, UIWindowSceneDelegate {

    var window: UIWindow?

    func scene(_ scene: UIScene, willConnectTo session: UISceneSession, options connectionOptions: UIScene.ConnectionOptions) {
        guard let windowScene = scene as? UIWindowScene else { return }

        // Disable overscan compensation to use full projector resolution (removes black borders)
        windowScene.screen.overscanCompensation = .none

        // Select the highest available resolution mode for the external display
        if let preferredMode = windowScene.screen.preferredMode {
            windowScene.screen.currentMode = preferredMode
        }

        let window = UIWindow(windowScene: windowScene)
        // Use UIHostingController to host the SwiftUI ContentView (Full Screen Visualizer)
        // ContentView observes the selectedVisualizer from VisualizerManager
        window.rootViewController = UIHostingController(rootView: ContentView())
        window.makeKeyAndVisible()

        self.window = window

        // Notify ExternalDisplayManager with the actual screen details
        ExternalDisplayManager.shared.externalDisplayDidConnect(
            screen: windowScene.screen,
            window: window
        )
    }

    func sceneDidDisconnect(_ scene: UIScene) {
        ExternalDisplayManager.shared.externalDisplayDidDisconnect()
        self.window = nil
    }
}
