struct SplatPLYConstants {
    enum ElementName: String {
        case point = "vertex"
    }

    enum PropertyName {
        static let positionX = [ "x" ]
        static let positionY = [ "y" ]
        static let positionZ = [ "z" ]
        static let normalX = [ "nx" ]
        static let normalY = [ "ny" ]
        static let normalZ = [ "nz" ]
        static let sh0_r = [ "f_dc_0" ]
        static let sh0_g = [ "f_dc_1" ]
        static let sh0_b = [ "f_dc_2" ]
        static let sphericalHarmonicsPrefix = "f_rest_"
        static let colorR = [ "red" ]
        static let colorG = [ "green" ]
        static let colorB = [ "blue" ]
        static let scaleX = [ "scale_0" ]
        static let scaleY = [ "scale_1" ]
        static let scaleZ = [ "scale_2" ]
        static let opacity = [ "opacity" ]
        static let rotation0 = [ "rot_0" ]
        static let rotation1 = [ "rot_1" ]
        static let rotation2 = [ "rot_2" ]
        static let rotation3 = [ "rot_3" ]

        // MARK: Ref-Gaussian (2DGS) relightable material properties.
        // Note: Ref-Gaussian PLYs are 2D surfels with only scale_0 / scale_1 (scale_2 is absent).
        // The primary normal residual reuses normalX/Y/Z ("nx"/"ny"/"nz") above.
        static let reflectionStrength = [ "refl_strength" ]
        static let roughness = [ "roughness" ]
        static let specularTintR = [ "ori_color_0" ]
        static let specularTintG = [ "ori_color_1" ]
        static let specularTintB = [ "ori_color_2" ]
        static let normal2X = [ "nx2" ]
        static let normal2Y = [ "ny2" ]
        static let normal2Z = [ "nz2" ]
    }
}
