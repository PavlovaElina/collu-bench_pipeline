#lang racket
(define (longest strings)
  (cond [(null? strings) #f]
        [else (foldl (lambda (s best)
                       (if (> (string-length s) (string-length best)) s best))
                     (car strings) (cdr strings))]))
