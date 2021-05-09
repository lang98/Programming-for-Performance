use core::mem::swap;
use super::{checksum::Checksum, idea::Idea, package::Package, Event};
use crossbeam::channel::{Receiver, Sender};
use std::io::{stdout, Write};
use std::sync::{Arc, Mutex};

pub struct Student {
    id: usize,
    idea: Option<Idea>,
    pkgs: Vec<Package>,
    skipped_idea: bool,
    event_sender_new: Sender<Event>,
    event_recv_new: Receiver<Event>,
    event_sender_out: Sender<Event>,
    event_recv_out: Receiver<Event>,
    event_sender_dl: Sender<Event>,
    event_recv_dl: Receiver<Event>,
    idea_checksum: Checksum,
    pkg_checksum: Checksum,
}

impl Student {
    pub fn new(
        id: usize,
        event_sender_new: Sender<Event>,
        event_recv_new: Receiver<Event>,
        event_sender_out: Sender<Event>,
        event_recv_out: Receiver<Event>,
        event_sender_dl: Sender<Event>,
        event_recv_dl: Receiver<Event>,
    ) -> Self {
        Self {
            id,
            event_sender_new,
            event_recv_new,
            event_sender_out,
            event_recv_out,
            event_sender_dl,
            event_recv_dl,
            idea: None,
            pkgs: vec![],
            skipped_idea: false,
            idea_checksum: Checksum::default(),
            pkg_checksum: Checksum::default(),
        }
    }

    fn build_idea(
        &mut self,
    ) {
        if let Some(ref idea) = self.idea {
            // Can only build ideas if we have acquired sufficient packages
            let pkgs_required = idea.num_pkg_required;
            if pkgs_required <= self.pkgs.len() {

                // Update idea and package checksums
                // All of the packages used in the update are deleted, along with the idea
                self.idea_checksum.update(Checksum::with_sha256(&idea.name));
                let pkgs_used = self.pkgs.drain(0..pkgs_required).collect::<Vec<_>>();
                for pkg in pkgs_used.iter() {
                    self.pkg_checksum.update(Checksum::with_sha256(&pkg.name));
                }

                self.idea = None;
            }
        }
    }


    pub fn run(&mut self, idea_checksum: Arc<Mutex<Checksum>>, pkg_checksum: Arc<Mutex<Checksum>>) {
        loop {
            if let Ok(Event::NewIdea(idea)) = self.event_recv_new.try_recv() {
                // If the student is not working on an idea, then they will take the new idea
                // and attempt to build it. Otherwise, the idea is skipped.
                if self.idea.is_none() {
                    self.idea = Some(idea);
                    self.build_idea();
                } else {
                    self.event_sender_new.send(Event::NewIdea(idea)).unwrap();
                    self.skipped_idea = true;
                }
            }

            if let Ok(Event::DownloadComplete(pkg)) = self.event_recv_dl.try_recv() {
                // Getting a new package means the current idea may now be buildable, so the
                // student attempts to build it
                self.pkgs.push(pkg);
                self.build_idea();
            }

            if let Ok(Event::OutOfIdeas) = self.event_recv_out.try_recv() {
                // If an idea was skipped, it may still be in the event queue.
                // If the student has an unfinished idea, they have to finish it, since they
                // might be the last student remaining.
                // In both these cases, we can't terminate, so the termination event is
                // deferred ti the back of the queue.
                if self.skipped_idea || self.idea.is_some() {
                    self.event_sender_out.send(Event::OutOfIdeas).unwrap();
                    self.skipped_idea = false;
                } else {
                    // Any unused packages are returned to the queue upon termination
                    for pkg in self.pkgs.drain(..) {
                        self.event_sender_dl
                            .send(Event::DownloadComplete(pkg))
                            .unwrap();
                    }
                    
                    // Update checksum here
                    let mut idea_chk = Checksum::default();
                    swap(&mut idea_chk, &mut self.idea_checksum);
                    idea_checksum.lock().unwrap().update(idea_chk);
                    let mut pkg_chk = Checksum::default();
                    swap(&mut pkg_chk, &mut self.pkg_checksum);
                    pkg_checksum.lock().unwrap().update(pkg_chk);
                    return;
                }
            }
        }
    }
}
