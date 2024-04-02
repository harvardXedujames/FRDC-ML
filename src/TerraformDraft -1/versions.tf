#check if this is really needed

#upon research i have found a direct integration between tf and gcp
// this could be the integration, but need to research further if this is the actual one
terraform {
  required_version = ">= 1.0.0"

  required_providers {
    google = {
      source  = "hashicorp/google"
      version = "~> 4.0"
    }
  }
}

provider "google" {
  credentials = file("<YOUR-CREDENTIALS-FILE>.json")
  project     = "<YOUR-GCP-PROJECT-ID>"
  region      = "<YOUR-GCP-REGION>"
  zone        = "<YOUR-GCP-ZONE>"
}
