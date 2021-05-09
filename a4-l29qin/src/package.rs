use super::checksum::Checksum;
use super::Event;
use crossbeam::channel::Sender;
use std::sync::{Arc, Mutex};

pub struct Package {
    pub name: String,
}

pub struct PackageDownloader {
    packages: Vec<String>,
    pkg_start_idx: usize,
    num_pkgs: usize,
    event_sender: Sender<Event>,
}

impl PackageDownloader {
    pub fn new(file: String, pkg_start_idx: usize, num_pkgs: usize, event_sender: Sender<Event>) -> Self {
        Self {
            packages: file.lines().map(|x| x.into()).collect(),
            pkg_start_idx,
            num_pkgs,
            event_sender,
        }
    }

    pub fn run(&self, pkg_checksum: Arc<Mutex<Checksum>>) {
        // Generate a set of packages and place them into the event queue
        // Update the package checksum with each package name
        let mut checksum = Checksum::default();
        for i in 0..self.num_pkgs {
            let name = self.packages
                .iter()
                .cycle()
                .nth(self.pkg_start_idx + i)
                .unwrap()
                .to_owned();

            checksum.update(Checksum::with_sha256(&name));
            self.event_sender
                .send(Event::DownloadComplete(Package { name }))
                .unwrap();
        }
        pkg_checksum
                .lock()
                .unwrap()
                .update(checksum);
    }
}
