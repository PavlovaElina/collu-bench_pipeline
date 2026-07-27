#lang racket
(define (fix_spaces text)
  (regexp-replace* #px" +" text
                   (lambda (m) (if (> (string-length m) 2) "-" (make-string (string-length m) #\_)))))
