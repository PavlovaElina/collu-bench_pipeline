#lang racket
(define (add_elements arr k)
  (for/sum ([x (in-list (take arr k))] #:when (< (abs x) 100)) x))
