#lang racket
(define (split_words txt)
  (cond [(for/or ([c (in-string txt)]) (char-whitespace? c)) (string-split txt)]
        [(string-contains? txt ",") (string-split txt ",")]
        [else (for/sum ([c (in-string txt)]
                        #:when (and (char<=? #\a c #\z) (odd? (- (char->integer c) 97)))) 1)]))
