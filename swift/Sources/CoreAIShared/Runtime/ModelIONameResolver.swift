/// Shared helpers for discovering model input/output names by substring matching.
public enum ModelIONameResolver {
    /// Finds the first name containing "pixel" or "image" (case-insensitive).
    public static func findImageInputName(in names: [String]) -> String? {
        names.first {
            let l = $0.lowercased()
            return l.contains("pixel") || l.contains("image")
        }
    }

    /// Finds the first name containing "logit" but NOT "presence" (case-insensitive).
    public static func findLogitsOutputName(in names: [String]) -> String? {
        names.first {
            let l = $0.lowercased()
            return l.contains("logit") && !l.contains("presence")
        }
    }

    /// Finds the first name containing "box" (case-insensitive).
    public static func findBoxesOutputName(in names: [String]) -> String? {
        names.first { $0.lowercased().contains("box") }
    }
}
