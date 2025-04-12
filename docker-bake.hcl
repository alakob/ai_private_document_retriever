# Default target group
# Use `docker buildx bake` to build all targets
# Use `docker buildx bake app` to build only the app target
group "default" {
  targets = ["app"]
}

# Definition for the main application image
target "app" {
  # Use the current directory as the build context
  context = "."
  # Specify the Dockerfile to use
  dockerfile = "Dockerfile"
  # Define the default tag for the image
  tags = ["docuseek-ai-app:latest"]
  # Load the image into the local Docker daemon
  output = ["type=docker"]
  # Optionally add build arguments here if needed in the future
  # args = { KEY = "value" }
  # Optionally define target platforms
  # platforms = ["linux/amd64", "linux/arm64"]
}

# Example of another potential target (e.g., a development version)
# target "app-dev" {
#   inherits = ["app"] # Inherit settings from the "app" target
#   dockerfile = "Dockerfile.dev" # Use a different Dockerfile
#   tags = ["docuseek-ai-app:dev"]
#   args = { BUILD_MODE = "development" }
# } 