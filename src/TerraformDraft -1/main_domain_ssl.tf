# Networking Configuration
// We disable auto-creation of subnetworks to have more control.
// A new subnet labelstudio_subnet is created in the VPC.
// An additional firewall rule firewall_ssh is added to allow SSH access, which is useful for initial setup and debugging. 
// Ensure you limit the source_ranges or use it temporarily for security.

resource "google_compute_network" "vpc_network" {
  name                    = "labelstudio-network"
  auto_create_subnetworks = false // Change this to false to manually create subnets
}

resource "google_compute_subnetwork" "labelstudio_subnet" {
  name          = "labelstudio-subnet"
  region        = var.region
  network       = google_compute_network.vpc_network.name
  ip_cidr_range = "10.0.1.0/24"
}

resource "google_compute_firewall" "firewall_ssh" {
  name    = "allow-ssh"
  network = google_compute_network.vpc_network.name

  allow {
    protocol = "tcp"
    ports    = ["22"]
  }

  source_ranges = ["0.0.0.0/0"]
}

# Additional security concerns from chatgpt, more will be addressed w the firewall tf -- link to the firewall.tf
# Restrict SSH Access: The SSH firewall rule is open to the world (0.0.0.0/0). Consider limiting it to your IP address or range.
# Subnet Design: The example subnet uses a /24 range in the 10.0.1.0 network. Adjust this based on your needs and IP planning.
# IAM Permissions: Ensure that your Terraform service account has minimal permissions required to create and manage these resources.


#domain and ssl stuffs
resource "google_dns_managed_zone" "labelstudio_zone" {
  name     = "labelstudio-zone"
  dns_name = "yourdomain.com."
}

resource "google_compute_managed_ssl_certificate" "labelstudio_ssl" {
  name        = "labelstudio-ssl"
  managed {
    domains = ["yourdomain.com"] # Replace yourdomain.com with your actual domain. 
  }
}

# Ensure your domain's nameservers are pointed to the Google Cloud DNS nameservers provided by the google_dns_managed_zone resource.

resource "google_compute_global_forwarding_rule" "https" {
  name       = "labelstudio-https-forwarding-rule"
  target     = google_compute_target_https_proxy.labelstudio_proxy.self_link
  port_range = "443"
  ip_address = google_compute_global_address.labelstudio_ip.address
}


#monitoring and logging
# This will create a logging sink that sends logs from your Label Studio VM to a BigQuery dataset for analysis.
resource "google_logging_project_sink" "labelstudio_logging" {
  name        = "labelstudio-logs"
  destination = "bigquery.googleapis.com/projects/${var.project_id}/datasets/labelstudio_logs" #Replace ${var.project_id} with your actual project ID.
  filter      = "resource.type=gce_instance AND resource.labels.instance_id=${google_compute_instance.labelstudio_vm.id}"
}
