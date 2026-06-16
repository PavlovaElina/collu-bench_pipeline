#lang racket
(define (Strongest_Extension class_name extensions)
  (define (strength ext)
    (- (for/sum ([c (in-string ext)] #:when (char-upper-case? c)) 1)
       (for/sum ([c (in-string ext)] #:when (char-lower-case? c)) 1)))
  (define best (foldl (lambda (ext acc) (if (> (strength ext) (strength acc)) ext acc))
                      (first extensions) (rest extensions)))
  (string-append class_name "." best))
