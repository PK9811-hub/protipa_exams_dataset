# 📝 To-Do List: Paper Revision (Based on EACL 2027 Reviewer Feedback)

## Top tier TODOs
- Fix task types in paper (open-ended, closed, structured) - to be added in Appendix [X]
- Check inpsect-ai logs in pgx1 [X]
- Check Penny's access to pgx1 (/home/shared/acl_2027/inspect-ai/) [X]
- Change train/dev split on Pan-Ex HF dataset to test/dev [X]
- Change 'test' split in py scripts and notebooks [X]
- Edit and enhance results tables with mistral judge scores [X]
- Add statistics table (correlation score etc) [X]
- Check latex issues in pass-or-fail-science subtask on Argilla [X]



## 1. Ανθρώπινη Αξιολόγηση & Επικύρωση Μετρικών 
- [X] **Αντιμετώπιση του Single-Judge Bias:** Να προστεθεί ανάλυση ανθεκτικότητας (robustness) του κριτή (π.χ. χρήση πολλαπλών judges αντί μόνο του Gemma-3-27B) για να αποδειχθεί ότι τα συμπεράσματα δεν βασίζονται σε ένα μεμονωμένο μοντέλο.
- [ ] **Αιτιολόγηση Επιλογής:** Να προστεθεί σαφής αιτιολόγηση για τον λόγο που επιλέχθηκε συγκεκριμένα το `Gemma-3-27B` ως evaluator.
- [X] **Επικύρωση LLM-as-a-Judge:** Να στηθεί ένα subset δεδομένων για αξιολόγηση από ανθρώπους (expert-human validation) προκειμένου να μετρηθεί η συμφωνία (inter-rater agreement/correlation) μεταξύ των βαθμολογιών του LLM-judge, του BERTScore και των ανθρώπων.

## 2. Στατιστική Ανάλυση & Μεθοδολογική Ακρίβεια
- [ ] **Significance Tests:** Να προστεθούν statistical significance tests ή/και διαστήματα εμπιστοσύνης (uncertainty estimates) για τις συγκρίσεις απόδοσης μεταξύ των μοντέλων.
- [X] **Προσαρμογή Ορολογίας:** Να αντικατασταθούν υπερβολικά ισχυρές εκφράσεις (π.χ., "proves", "significantly outperforms") με πιο προσεκτική, αιτιακή ακαδημαϊκή ορολογία.
- [ ] **Διαχωρισμός Αθροιστικών Μετρικών:** Να επανεξεταστεί ή να διευκρινιστεί η προσέγγιση της σύγκρισης συγκεντρωτικών ποσοστών (aggregate percentages) όταν αναμειγνύονται διαφορετικές μέθοδοι βαθμολόγησης (exact match, custom normalized, LLM grading).
- [ ] **Ablation Studies για το "Few-shot Paradox":** Να διεξαχθούν πειράματα αφαίρεσης (ablations) για να ξεκαθαριστεί αν η πτώση απόδοσης οφείλεται σε υπερφόρτωση του context window (context-length overload) ή σε σύγχυση λόγω της δομής του prompt (prompt-format confusion).

## 3. Αναδιατύπωση Novelty & Βιβλιογραφία
- [X] **Προσθήκη Σχετικού Dataset:** Να προστεθεί παραπομπή στο υπάρχον dataset `mpvasilis/thema-panhellenic-exams` στο Hugging Face.
- [X] **Αναπλαισίωση Ευρημάτων:** Να αναγνωριστεί ρητά ότι συγκεκριμένα ευρήματα της έρευνας αποτελούν επιβεβαίωση γνωστών φαινομένων στο νέο (Ελληνικό) περιβάλλον και όχι εντελώς νέες ανακαλύψεις. Συγκεκριμένα:
  - [X] H αξιολόγηση LLMs σε multi-subject tasks (π.χ. MMLU, Hendrycks et al., 2021).
  - [X] Οι περιορισμοί του BERTScore στα open-ended (He et al., ACL 2023).
  - [X] Η ανταγωνιστικότητα μικρότερων adapted μοντέλων (π.χ. KriKri) έναντι μεγαλύτερων (Nguyen et al., ACL 2024).
  - [X] Η τάση επιείκειας (leniency) των LLM-as-a-Judge (Thakur et al.).

## 4. Πειραματικές Επεκτάσεις (Experimental Setup)
- [ ] **Chain-of-Thought (CoT):** Να διερευνηθεί και να ενσωματωθεί η μέθοδος Chain-of-Thought στην αξιολόγηση.
- [ ] **Ποιοτική Ανάλυση Λογικής (Reasoning Analysis):** Να γίνει ανάλυση των βημάτων λογικής (reasoning steps) που ακολουθούν τα μοντέλα σε συγκεκριμένα προβλήματα/tasks.
- [ ] **Περισσότερα Παραδείγματα (Qualitative Examples):** Να προστεθούν συγκεκριμένα παραδείγματα/case studies που να δείχνουν στον αναγνώστη πού υπερτερεί και πού υστερεί το κάθε μοντέλο. 

## 5. Dataset Περιορισμοί & Πηγές
- [x] **Προσθήκη Πηγών (Source Links):** Να προστεθούν τα αρχικά links από όπου κατέβηκαν τα επίσημα θέματα των εξετάσεων.
- [X] **Περιορισμοί Μεγέθους (Size Limitations):** Να αναδειχθεί με διαφάνεια ως περιορισμός το μικρό μέγεθος του private Prot-Ex set (ιδίως σε συγκεκριμένα μαθήματα όπως η Φυσική, n=9), και πώς αυτό επηρεάζει τη γενίκευση (generalization) των συμπερασμάτων.