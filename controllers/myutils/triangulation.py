class Triangulation:
    def __init__(self, knownDistance, knownWidth, referenceArea):
        self.focalLength = referenceArea * knownDistance / knownWidth
        self.knownDistance = knownDistance
        self.knownWidth = knownWidth
        pass

    def distance_to_camera(self, area):
        return self.knownWidth * self.focalLength / area
    