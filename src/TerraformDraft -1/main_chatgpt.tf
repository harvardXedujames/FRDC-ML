#from chatgpt
resource "google_compute_instance" "label_studio_instance" {
  name         = "label-studio"
  machine_type = "e2-medium"  # Adjust the machine type as needed

  boot_disk {
    initialize_params {
      image = "ubuntu-os-cloud/ubuntu-2004-lts"   #what is this?
    }
  }

  network_interface {
    network = "default"
    access_config {
      // Ephemeral IP
    }
  }

  metadata = {
    ssh-keys = "ubuntu:${file("<YOUR_SSH_PUBLIC_KEY>.pub")}"
  }

  service_account {
    scopes = ["https://www.googleapis.com/auth/cloud-platform"]
  }

  provisioner "remote-exec" {
    inline = [
      "sudo apt-get update",
      "sudo apt-get install -y docker.io docker-compose",
      "sudo systemctl start docker",
      "sudo systemctl enable docker",
      "sudo usermod -aG docker $USER",
      "docker run -d -p 8080:8080 -v ~/label-studio:/label-studio heartexlabs/label-studio:latest"
    ]

    connection {
      type        = "ssh"
      user        = "ubuntu"
      private_key = file("<YOUR_SSH_PRIVATE_KEY>")
      host        = self.network_interface[0].access_config[0].nat_ip
    }
  }
}
