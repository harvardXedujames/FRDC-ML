# GCP Provider

provider "google"{
    credentials = file {var.gcp_svc_key}
    project = var.gcp_project # assign variable values and not hardcoded
    region = var.gcp_region
}

provider "google" {
  credentials = file("<YOUR_SERVICE_ACCOUNT_KEY_FILE>.json")
  project     = "<YOUR_PROJECT_ID>"
  region      = "  "  # Change this to your preferred region
  zone        = " " # Change this to your preferred zone
}
