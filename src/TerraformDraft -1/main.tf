
# All about data
# create cloud stroage bucket
resource "google_storage_bucket" "website " {
    provider = "" # need to fill in
    name = "example"  #name needs to be globally unqiue as per gcp
    location = "" # okay not sure what is this, could be region too 
}

#ideal data solution
resource "google_storage_bucket" "labelstudio_data" {
  name                        = "labelstudio-${var.project_id}-data"    #Replace ${var.project_id} with your actual GCP project ID or another unique identifier.
  location                    = var.region                              #The location should match the region where you're deploying Label Studio for better performance.
  storage_class               = "STANDARD"                              #storage_class is set to "STANDARD" for general-purpose storage.
  uniform_bucket_level_access = true                                    #uniform_bucket_level_access is enabled for consistent IAM policy management.

    #A lifecycle rule is added to automatically delete objects older than one year. Adjust this based on your data retention policy.
  lifecycle_rule {
    condition {
      age = 365
    }
    action {
      type = "Delete"
    }
  }
}

#----------------

# make new objects public, maybe not good for our project
resource "google_storage_bucket_access_control" "name" {
    object = 
    bucket = 
    role = "reader"
    entity = "allUsers" #figure out how do i change it from all users to restricted only
}

#upload html file to the bucket
resource "google_storage_bucket_object" "static_site_src" {
    name = "index.html"
    source = "root directory/website/index.html" #defime source - where the object is loaded to
    bucket = google_storage_bucket.website.name
}
#-----------------

#ideally (main decalaration + firewall, for firewall look at the -- firewall.tf, that might be better)
resource "google_compute_instance" "labelstudio_vm" {
  name         = "labelstudio-vm"
  machine_type = "e2-medium"
  zone         = var.zone

  boot_disk {
    initialize_params {
      image = "debian-cloud/debian-9" #We define a VM instance labelstudio_vm with Debian 9 image.
    }
  }

 # The metadata_startup_script refers to a shell script startup-script.sh that will install Docker and run Label Studio. You need to create this script.
  network_interface {
    network = google_compute_network.vpc_network.self_link
    access_config {
      // Ephemeral IP
    }
  }

  metadata_startup_script = file("startup-script.sh")
}

# The google_compute_network.vpc_network resource creates a VPC network for our VM.
resource "google_compute_network" "vpc_network" {
  name                    = "labelstudio-network"
  auto_create_subnetworks = true
}

# Two google_compute_firewall resources allow HTTP and HTTPS traffic to the VM.
resource "google_compute_firewall" "firewall_http" {
  name    = "allow-http"
  network = google_compute_network.vpc_network.name

  allow {
    protocol = "tcp"
    ports    = ["80"]
  }

  source_ranges = ["0.0.0.0/0"]
}

resource "google_compute_firewall" "firewall_https" {
  name    = "allow-https"
  network = google_compute_network.vpc_network.name

  allow {
    protocol = "tcp"
    ports    = ["443"]
  }

  source_ranges = ["0.0.0.0/0"]
}

#-----------

