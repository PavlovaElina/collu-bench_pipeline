#lang racket
(define (maximum arr k)
  (if (= k 0) '()
      (sort (take (sort arr >) k) <)))
