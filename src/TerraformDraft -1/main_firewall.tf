resource "google_compute_firewall" "allow_http" {
  name    = "allow-http"
  network = "default"

  allow {
    protocol = "tcp"
    ports    = ["80"]
  }

  source_ranges = ["0.0.0.0/0"] # Be cautious with this, it allows traffic from any source
  target_tags   = ["http-server"] # Apply this rule to instances tagged with 'http-server'
}

resource "google_compute_firewall" "allow_https" {
  name    = "allow-https"
  network = "default"

  allow {
    protocol = "tcp"
    ports    = ["443"]
  }

  source_ranges = ["0.0.0.0/0"] # Be cautious with this, it allows traffic from any source
  target_tags   = ["https-server"] # Apply this rule to instances tagged with 'https-server'
}

resource "google_compute_firewall" "allow_label_studio" {
  name    = "allow-label-studio"
  network = "default"

  allow {
    protocol = "tcp"
    ports    = ["8080"] # Default port for Label Studio
  }

  source_ranges = ["0.0.0.0/0"] # Adjust this to restrict access to specific IPs
  target_tags   = ["label-studio-server"] # Apply this rule to instances tagged with 'label-studio-server', this tag will be used to identify VMs that this rule applies to
}


//  blah blah.allow_http and  blah blah.allow_https are rules to allow HTTP and HTTPS traffic, respectively. You might need these for web services running on your instances.
// " blah blah.allow_label_studio is specifically set up for Label Studio, which typically runs on port 8080.\

#need to tag the firewall rules to the gcp this should be under main.tf
resource "google_compute_instance" "label_studio_instance" {
  tags = ["http-server", "https-server", "label-studio-server"] # Add this line
}
